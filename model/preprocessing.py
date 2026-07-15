import random
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance


class SyncedAugment:
    def __init__(self, image_size=(512, 512), hflip=0.5, vflip=0.2, rotation=15, color_jitter=True):
        self.image_size = image_size
        self.hflip = hflip
        self.vflip = vflip
        self.rotation = rotation
        self.color_jitter = color_jitter

    def __call__(self, rgb, ela, mask):
        if random.random() < self.hflip:
            rgb  = rgb.transpose(Image.FLIP_LEFT_RIGHT)
            ela  = ela.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < self.vflip:
            rgb  = rgb.transpose(Image.FLIP_TOP_BOTTOM)
            ela  = ela.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        if self.rotation > 0 and random.random() < 0.4:
            angle = random.uniform(-self.rotation, self.rotation)
            rgb  = rgb.rotate(angle, resample=Image.BILINEAR, fillcolor=(245, 245, 245))
            ela  = ela.rotate(angle, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
            mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        if random.random() < 0.3:
            scale = random.uniform(0.85, 1.0)
            new_w = int(self.image_size[0] * scale)
            new_h = int(self.image_size[1] * scale)
            left  = random.randint(0, self.image_size[0] - new_w)
            top   = random.randint(0, self.image_size[1] - new_h)
            rgb  = rgb.crop((left, top, left + new_w, top + new_h)).resize(self.image_size, Image.BILINEAR)
            ela  = ela.crop((left, top, left + new_w, top + new_h)).resize(self.image_size, Image.BILINEAR)
            mask = mask.crop((left, top, left + new_w, top + new_h)).resize(self.image_size, Image.NEAREST)

        if self.color_jitter and random.random() < 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast   = random.uniform(0.8, 1.2)
            rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
            rgb = ImageEnhance.Contrast(rgb).enhance(contrast)

        if random.random() < 0.2:
            rgb = rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

        return rgb, ela, mask


class RGBTransform:
    def __init__(self, image_size=(512, 512)):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img)


class ELATransform:
    def __init__(self, image_size=(512, 512)):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        return self.transform(img)


class MaskTransform:
    def __init__(self, image_size=(512, 512)):
        self.resize = transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, mask):
        mask = self.resize(mask)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()
        return mask


def get_train_transforms(image_size=(512, 512)):
    synced = SyncedAugment(image_size=image_size, hflip=0.5, vflip=0.2, rotation=15, color_jitter=True)
    rgb_tf  = RGBTransform(image_size)
    ela_tf  = ELATransform(image_size)
    mask_tf = MaskTransform(image_size)
    return synced, rgb_tf, ela_tf, mask_tf


def get_val_transforms(image_size=(512, 512)):
    rgb_tf  = RGBTransform(image_size)
    ela_tf  = ELATransform(image_size)
    mask_tf = MaskTransform(image_size)
    return None, rgb_tf, ela_tf, mask_tf


def normalize_ela(ela_tensor):
    mn = ela_tensor.min()
    mx = ela_tensor.max()
    if mx - mn < 1e-6:
        return ela_tensor
    return (ela_tensor - mn) / (mx - mn)


def compute_class_weights(dataset):
    forged    = 0
    authentic = 0
    for idx in range(len(dataset)):
        _, _, mask = dataset[idx]
        if mask.sum() > 0:
            forged += 1
        else:
            authentic += 1
    total = forged + authentic
    w_forged    = total / (2 * forged)    if forged    > 0 else 1.0
    w_authentic = total / (2 * authentic) if authentic > 0 else 1.0
    weights = []
    for idx in range(len(dataset)):
        _, _, mask = dataset[idx]
        weights.append(w_forged if mask.sum() > 0 else w_authentic)
    return weights


if __name__ == '__main__':
    from PIL import Image as PILImage
    import torch

    rgb_pil  = PILImage.new('RGB', (512, 512), (240, 240, 240))
    ela_pil  = PILImage.new('RGB', (512, 512), (10, 10, 10))
    mask_pil = PILImage.new('L',   (512, 512), 0)

    synced, rgb_tf, ela_tf, mask_tf = get_train_transforms()
    rgb_aug, ela_aug, mask_aug = synced(rgb_pil, ela_pil, mask_pil)

    rgb_t  = rgb_tf(rgb_aug)
    ela_t  = ela_tf(ela_aug)
    mask_t = mask_tf(mask_aug)

    print(f"RGB tensor shape  : {rgb_t.shape}")
    print(f"ELA tensor shape  : {ela_t.shape}")
    print(f"Mask tensor shape : {mask_t.shape}")
    print(f"Mask unique       : {mask_t.unique().tolist()}")
    print(f"RGB  min/max      : {rgb_t.min():.3f} / {rgb_t.max():.3f}")
    print(f"ELA  min/max      : {ela_t.min():.3f} / {ela_t.max():.3f}")
    print("Preprocessing OK.")
