import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance

def get_splits(dataset, train_frac=0.70, val_frac=0.15):
    total      = len(dataset)
    train_size = int(train_frac * total)
    val_size   = int(val_frac * total)
    test_size  = total - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])

class ForgeryDataset(Dataset):
    # Just a simple dataset for all our forgery data
    def __init__(self, data_root=None, dataset_names=None, crop_size=256, is_train=True, image_dir=None, mask_dir=None):
        self.data_root = data_root
        self.crop_size = crop_size
        self.is_train = is_train
        self.samples = []
        
        # Support for test_pipeline.py style (direct directories)
        if image_dir and mask_dir:
            if os.path.exists(image_dir):
                all_files = os.listdir(image_dir)
                for f in sorted(all_files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                        continue
                    stem = os.path.splitext(f)[0]
                    mask_candidates = [
                        os.path.join(mask_dir, stem + '_gt.png'),
                        os.path.join(mask_dir, stem + '.png'),
                        os.path.join(mask_dir, stem + '.jpg'),
                        os.path.join(mask_dir, stem + '_mask.png'),
                    ]
                    mask_path = next((m for m in mask_candidates if os.path.exists(m)), None)
                    if mask_path:
                        self.samples.append((os.path.join(image_dir, f), mask_path))
            print(f"Loaded {len(self.samples)} samples from {image_dir}")
        # Standard usage from train.py
        elif data_root and dataset_names:
            for dname in dataset_names:
                path = os.path.join(data_root, dname)
                if not os.path.exists(path):
                    print(f"Warning: {path} not found!")
                    continue
                    
                # assuming all images are jpg and masks are png
                images = glob.glob(os.path.join(path, '*.jpg'))
                for img_path in sorted(images):
                    # masks usually have _mask appended before extension
                    mask_path = img_path.replace('.jpg', '_mask.png')
                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))
                    else:
                        # try without _mask just in case
                        mask_path2 = img_path.replace('.jpg', '.png')
                        if os.path.exists(mask_path2):
                            self.samples.append((img_path, mask_path2))
            print(f"Loaded {len(self.samples)} samples from {dataset_names}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]
        
        # load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # resize mask to image size if they don't match for some reason
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # make mask binary
        mask = (mask > 127).astype(np.uint8)
        
        # get edge map via Canny for a thinner boundary
        edge = cv2.Canny(mask * 255, 100, 200)
        edge = (edge > 0).astype(np.uint8)
        
        # Convert to PIL for synced augmentations
        img_pil = Image.fromarray(img)
        mask_pil = Image.fromarray(mask * 255)
        edge_pil = Image.fromarray(edge * 255)
        
        if self.is_train:
            # SyncedAugment logic
            if random.random() < 0.5: # hflip
                img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
                edge_pil = edge_pil.transpose(Image.FLIP_LEFT_RIGHT)

            if random.random() < 0.2: # vflip
                img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
                mask_pil = mask_pil.transpose(Image.FLIP_TOP_BOTTOM)
                edge_pil = edge_pil.transpose(Image.FLIP_TOP_BOTTOM)

            if random.random() < 0.4: # rotation (15 deg)
                angle = random.uniform(-15, 15)
                img_pil = img_pil.rotate(angle, resample=Image.BILINEAR, fillcolor=(245, 245, 245))
                mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)
                edge_pil = edge_pil.rotate(angle, resample=Image.NEAREST, fillcolor=0)

            if random.random() < 0.3: # random crop and scale
                scale = random.uniform(0.85, 1.0)
                new_w = int(img_pil.width * scale)
                new_h = int(img_pil.height * scale)
                left = random.randint(0, max(0, img_pil.width - new_w))
                top = random.randint(0, max(0, img_pil.height - new_h))
                img_pil = img_pil.crop((left, top, left + new_w, top + new_h))
                mask_pil = mask_pil.crop((left, top, left + new_w, top + new_h))
                edge_pil = edge_pil.crop((left, top, left + new_w, top + new_h))

            # Resize to crop_size
            img_pil = img_pil.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            mask_pil = mask_pil.resize((self.crop_size, self.crop_size), Image.NEAREST)
            edge_pil = edge_pil.resize((self.crop_size, self.crop_size), Image.NEAREST)

            if random.random() < 0.5: # color jitter
                brightness = random.uniform(0.8, 1.2)
                contrast = random.uniform(0.8, 1.2)
                img_pil = ImageEnhance.Brightness(img_pil).enhance(brightness)
                img_pil = ImageEnhance.Contrast(img_pil).enhance(contrast)

            if random.random() < 0.2: # gaussian blur
                img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
                
        else:
            # just resize for test
            img_pil = img_pil.resize((self.crop_size, self.crop_size), Image.BILINEAR)
            mask_pil = mask_pil.resize((self.crop_size, self.crop_size), Image.NEAREST)
            edge_pil = edge_pil.resize((self.crop_size, self.crop_size), Image.NEAREST)

        # Convert back to numpy
        img = np.array(img_pil)
        mask = np.array(mask_pil)
        edge = np.array(edge_pil)
        
        mask = (mask > 127).astype(np.uint8)
        edge = (edge > 127).astype(np.uint8)
        
        # convert to torch tensors [C, H, W]
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
        edge_tensor = torch.from_numpy(edge).float().unsqueeze(0)
        
        return img_tensor, mask_tensor, edge_tensor

def get_dataloader(dataset_names, batch_size=4, is_train=True, return_splits=False):
    # wrapper to get the dataloader easily
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
        
    ds = ForgeryDataset('data_pipeline/raw', dataset_names, is_train=is_train)
    
    if return_splits:
        total = len(ds)
        train_size = int(0.8 * total)
        val_size = int(0.1 * total)
        test_size = total - train_size - val_size
        
        train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])
        
        # NOTE: Ideally val and test shouldn't have augmentations. For now we use the same dataset object.
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader, test_loader
        
    return DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=0)

if __name__ == '__main__':
    # test script to make sure it works and generate progress update
    datasets = ['CASIAv2', 'DocTamper', 'Defacto', 'MIDV-500']
    
    # load a small batch of 4
    loader = get_dataloader(datasets, batch_size=4)
    images, masks, edges = next(iter(loader))
    
    os.makedirs('reports', exist_ok=True)
    
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i in range(images.shape[0]):
        # tensor to numpy
        img = images[i].permute(1, 2, 0).numpy()
        mask = masks[i].squeeze(0).numpy()
        edge = edges[i].squeeze(0).numpy()
        
        # make overlay (red tint on mask)
        overlay = img.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Sample {i} Image')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(overlay)
        axes[1, i].set_title(f'Mask Overlay')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(edge, cmap='gray')
        axes[2, i].set_title(f'Edge Map')
        axes[2, i].axis('off')
        
    plt.tight_layout()
    plt.savefig('reports/dataset_samples.png')
    print("Saved progress image to reports/dataset_samples.png")
