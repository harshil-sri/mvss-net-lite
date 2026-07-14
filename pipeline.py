import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

from torchvision import transforms
from PIL import Image, ImageChops, ImageEnhance


def generate_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')
    temp_path = 'temp_ela.jpg'
    original.save(temp_path, 'JPEG', quality=quality)
    compressed = Image.open(temp_path).convert('RGB')
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image


def get_splits(dataset, train_frac=0.70, val_frac=0.15):
    total      = len(dataset)
    train_size = int(train_frac * total)
    val_size   = int(val_frac * total)
    test_size  = total - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size])


class ForgeryDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(512, 512), max_samples=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size

        all_files = []
        for f in os.listdir(image_dir):
            ext = os.path.splitext(f)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png', '.tif', '.tiff'):
                continue
            stem = os.path.splitext(f)[0]
            mask_candidates = [
                os.path.join(mask_dir, stem + '_gt.png'),
                os.path.join(mask_dir, stem + '.png'),
            ]
            if any(os.path.exists(m) for m in mask_candidates):
                all_files.append(f)
        self.image_files = sorted(all_files)[:max_samples] if max_samples else sorted(all_files)

        self.rgb_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.ela_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        stem = os.path.splitext(image_name)[0]
        mask_candidates = [
            os.path.join(self.mask_dir, stem + '_gt.png'),
            os.path.join(self.mask_dir, stem + '.png'),
        ]
        mask_path = next(m for m in mask_candidates if os.path.exists(m))

        rgb_image = Image.open(image_path).convert('RGB')
        ela_image = generate_ela(image_path)
        mask = Image.open(mask_path).convert('L')

        rgb_tensor = self.rgb_transform(rgb_image)
        ela_tensor = self.ela_transform(ela_image)
        mask_tensor = self.mask_transform(mask)
        mask_tensor = (mask_tensor > 0.5).float()

        return rgb_tensor, ela_tensor, mask_tensor


if __name__ == '__main__':
    image_dir = 'data/doctamper/images'
    mask_dir  = 'data/doctamper/masks'

    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        print(f"Warning: directories not found — {image_dir}, {mask_dir}")
        sys.exit(1)

    dataset = ForgeryDataset(image_dir=image_dir, mask_dir=mask_dir)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    rgb, ela, mask = next(iter(dataloader))
    print(f"RGB shape: {rgb.shape}")
    print(f"ELA shape: {ela.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask unique values: {mask.unique()}")
    print("Pipeline working.")

    total = len(dataset)
    train_size = int(0.7 * total)
    val_size   = int(0.15 * total)
    test_size  = total - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    weights = []
    for idx in train_set.indices:
        mask_name = dataset.image_files[idx].replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(mask_dir, mask_name)
        mask_img = Image.open(mask_path).convert('L')
        is_forged = 1 if any(mask_img.tobytes()) else 0
        weights.append(is_forged)

    n_forged    = sum(weights)
    n_authentic = len(weights) - n_forged
    class_weights = [1.0 / n_forged if w == 1 else 1.0 / max(n_authentic, 1) for w in weights]

    sampler = WeightedRandomSampler(class_weights, num_samples=len(class_weights), replacement=True)
    balanced_loader = DataLoader(train_set, batch_size=4, sampler=sampler, num_workers=2)

    print(f"Forged samples: {n_forged}, Authentic samples: {n_authentic}")
    print("Weighted sampler active — balanced batches enabled.")