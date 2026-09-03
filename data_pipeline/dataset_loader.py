import os
import glob
import random
import cv2
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance
from torch.utils.data import WeightedRandomSampler

class ForgeryDataset(Dataset):
    def __init__(self, samples, crop_size=256, is_train=True):
        self.crop_size = crop_size
        self.is_train = is_train
        self.samples = samples
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        mask = (mask > 127).astype(np.uint8)
        
        edge = cv2.Canny(mask * 255, 100, 200)
        edge = (edge > 0).astype(np.uint8)
        
        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray(mask * 255)
        edge_pil = Image.fromarray(edge * 255)
        
        if self.is_train:
            if random.random() < 0.5:
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
                edge_pil = edge_pil.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.2:
                img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)
                edge_pil = edge_pil.transpose(Image.FLIP_TOP_BOTTOM)
            if random.random() < 0.4:
                angle = random.uniform(-15, 15)
                img_pil = img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=(245, 245, 245))
                mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
                edge_pil = edge_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
            if random.random() < 0.3:
                scale = random.uniform(0.85, 1.0)
                new_w = int(img_pil.width * scale)
                new_h = int(img_pil.height * scale)
                left = random.randint(0, max(0, img_pil.width - new_w))
                top = random.randint(0, max(0, img_pil.height - new_h))
                img_pil = img_pil.crop((left, top, left + new_w, top + new_h))
                mask_pil = mask_pil.crop((left, top, left + new_w, top + new_h))
                edge_pil = edge_pil.crop((left, top, left + new_w, top + new_h))

            import torch.nn.functional as F
            img_pil = img_pil.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            mask_t = torch.from_numpy(np.array(mask_pil)).float().unsqueeze(0).unsqueeze(0)
            mask_pil = Image.fromarray(F.adaptive_max_pool2d(mask_t, (self.crop_size, self.crop_size)).squeeze().numpy().astype(np.uint8))
            edge_t = torch.from_numpy(np.array(edge_pil)).float().unsqueeze(0).unsqueeze(0)
            edge_pil = Image.fromarray(F.adaptive_max_pool2d(edge_t, (self.crop_size, self.crop_size)).squeeze().numpy().astype(np.uint8))

            if random.random() < 0.5:
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
                img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)
            if random.random() < 0.2:
                img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
                
        else:
            import torch.nn.functional as F
            img_pil = img_pil.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            mask_t = torch.from_numpy(np.array(mask_pil)).float().unsqueeze(0).unsqueeze(0)
            mask_pil = Image.fromarray(F.adaptive_max_pool2d(mask_t, (self.crop_size, self.crop_size)).squeeze().numpy().astype(np.uint8))
            edge_t = torch.from_numpy(np.array(edge_pil)).float().unsqueeze(0).unsqueeze(0)
            edge_pil = Image.fromarray(F.adaptive_max_pool2d(edge_t, (self.crop_size, self.crop_size)).squeeze().numpy().astype(np.uint8))

        img = np.array(img_pil)
        mask = np.array(mask_pil)
        edge = np.array(edge_pil)
        
        mask = (mask > 127).astype(np.uint8)
        edge = (edge > 127).astype(np.uint8)
        
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        edge_tensor = torch.from_numpy(edge).float().unsqueeze(0)
        
        return img_tensor, mask_tensor, edge_tensor

# =============================================================================
# Sampler support: per-sample domain/forgery classes + dynamically measured
# balanced-sampler weights (Stage 2 Run 2).
#
# Target composition: 1/3 RTM-forged / 1/3 RTM-authentic / 1/3 MIDV500.
# Weights are computed from ACTUAL class counts measured off the manifest's
# train split at load time. The old hardcoded denominators (1/6000, 1/3000,
# 1/15050) did not match the real population and silently skewed composition.
# =============================================================================

TARGET_CLASSES = ('rtm_forged', 'rtm_auth', 'midv500_forged', 'midv500_auth')

def compute_sample_classes(samples):
    """
    Classifies each (img_path, mask_path) sample:
      'rtm_forged'  - RTM image whose mask has any positive pixel
      'rtm_auth'    - RTM image with an empty mask
      'midv500_forged' - MIDV500 image with a non-empty mask
      'midv500_auth'   - MIDV500 image with an empty mask
      'other'       - anything else
    """
    labels = []
    for img_path, mask_path in samples:
        is_rtm = 'RTM' in img_path
        is_midv = 'MIDV500' in img_path
        if is_rtm:
            is_forged = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).sum() > 0
            labels.append('rtm_forged' if is_forged else 'rtm_auth')
        elif is_midv:
            is_forged = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).sum() > 0
            labels.append('midv500_forged' if is_forged else 'midv500_auth')
        else:
            labels.append('other')
    return labels

def build_balanced_sampler_weights(labels):
    """
    Equal probability mass per present target class -> realized draws converge
    to exact 1/N_target composition by construction.
    Non-target samples get weight 0.0 (excluded from sampling); if no target
    class is present at all, falls back to uniform weights so generic
    (Stage-1-style) dataset mixes keep working unchanged.
    Returns (weights, class_counts_dict).
    """
    from collections import Counter
    counts = Counter(labels)
    present = {c: counts[c] for c in TARGET_CLASSES if counts.get(c, 0) > 0}
    if not present:
        print(f"WARNING: none of {TARGET_CLASSES} present in this split "
              f"(counts={dict(counts)}). Falling back to uniform sampler weights.")
        return [1.0] * len(labels), dict(counts)

    per_class_mass = 1.0 / len(present)
    weights = [per_class_mass / counts[lab] if lab in present else 0.0 for lab in labels]

    print("Balanced sampler: measured train-split class counts:")
    for c in TARGET_CLASSES:
        n = counts.get(c, 0)
        share = per_class_mass if c in present else 0.0
        print(f"  {c}: count={n} | target draw prob per sample={share}/{n if n else '-'} "
              f"| class mass={share:.4f}")
    other = sum(v for k, v in counts.items() if k not in TARGET_CLASSES)
    if other > 0:
        print(f"  WARNING: {other} non-target samples excluded from sampling (weight=0.0)")
    return weights, dict(counts)

class IndexedForgeryDataset(Dataset):
    """
    Wraps a ForgeryDataset; __getitem__ returns (img, mask, edge, idx) so the
    training loop can log realized per-batch sampler composition from batch one.
    """

    def __init__(self, base_ds):
        self.base = base_ds
        self.samples = base_ds.samples

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, mask, edge = self.base[idx]
        return img, mask, edge, idx


def _make_loader(ds, batch_size, is_train, use_balanced_sampler, return_index):
    """Builds a DataLoader with optional balanced sampler + index tracking."""
    labels = None
    sampler = None
    shuffle = is_train
    if is_train and (use_balanced_sampler or return_index):
        # Measuring classes reads every mask once; only do it when needed.
        labels = compute_sample_classes(ds.samples)
        ds.sample_classes = labels

    if is_train and use_balanced_sampler:
        shuffle = False
        weights, _ = build_balanced_sampler_weights(labels)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    inner = IndexedForgeryDataset(ds) if (is_train and return_index) else ds
    return DataLoader(inner, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0)


def get_dataloader(dataset_names, batch_size=4, is_train=True, return_splits=False,
                   use_balanced_sampler=False, return_index=False):
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
        
    manifest_path = 'reports/manifest.json'
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found at {manifest_path}. Please run build_manifest.py first.")
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    def filter_split(split_data):
        filtered = []
        for s in split_data:
            if s['dataset'] in dataset_names:
                filtered.append((s['image'], s['mask']))
        return filtered

    train_samples = filter_split(manifest['train'])
    val_samples = filter_split(manifest['val'])
    test_samples = filter_split(manifest['test'])
    
    if not train_samples and not val_samples and not test_samples:
        print(f"Warning: No samples found for datasets {dataset_names} in manifest!")
        
    train_ds = ForgeryDataset(train_samples, is_train=is_train)
    val_ds = ForgeryDataset(val_samples, is_train=False)
    test_ds = ForgeryDataset(test_samples, is_train=False)

    if return_splits:
        train_loader = _make_loader(train_ds, batch_size, is_train, use_balanced_sampler, return_index)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, test_loader

    # If not returning splits, behave like a single loader.
    # For backward compatibility, combine all requested splits into one, or just return train.
    # Usually is_train=True means we want the train loader, else we want val.
    target_samples = train_samples if is_train else val_samples
    ds = ForgeryDataset(target_samples, is_train=is_train)
    return _make_loader(ds, batch_size, is_train, use_balanced_sampler, return_index)

if __name__ == '__main__':
    datasets = ['CASIAv2', 'DEFACTO']
    loader = get_dataloader(datasets, batch_size=4, return_splits=False)
    images, masks, edges = next(iter(loader))
    print("Successfully loaded batch!")
