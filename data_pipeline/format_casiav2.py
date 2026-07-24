import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def format_casiav2(images_dir: str, masks_dir: str, output_dir: str):
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        for path in images_path.rglob(ext):
            if not path.stem.endswith('_gt'):
                image_files.append(path)
        for path in images_path.rglob(ext.upper()):
            if not path.stem.endswith('_gt'):
                image_files.append(path)
        
    print(f"Found {len(image_files)} total images in {images_dir}. Processing...")
    
    processed = 0
    tampered = 0
    authentic = 0
    
    for img_path in image_files:
        stem = img_path.stem
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        out_img_path = out_path / f"{stem}.jpg"
        cv2.imwrite(str(out_img_path), img)
        
        # CASIAv2 logic: Authentic images usually start with 'Au', Tampered with 'Tp'
        is_tampered = stem.startswith('Tp')
        
        mask_path = None
        if is_tampered and masks_path.exists():
            # Try to find ground truth mask. Common naming conventions:
            for suffix in ['', '_gt', '_mask']:
                for ext in ['.png', '.jpg', '.tif']:
                    possible_mask = masks_path / f"{stem}{suffix}{ext}"
                    if possible_mask.exists():
                        mask_path = possible_mask
                        break
                if mask_path: break

        out_mask_path = out_path / f"{stem}_mask.png"
        
        if mask_path:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                # Binarize and ensure correct size
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_mask_path), mask)
            tampered += 1
        else:
            # Generate zero mask for authentic (or if mask is missing for some reason)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(out_mask_path), mask)
            if not is_tampered:
                authentic += 1
            
        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed} images...")

    print(f"\\nCASIAv2 Formatting Done! Total Processed: {processed}")
    print(f"Tampered (mask found): {tampered}")
    print(f"Authentic (zero-mask generated): {authentic}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format CASIAv2 dataset")
    parser.add_argument("--images", type=str, default="data/CASIAv2_raw/images", help="Path to raw images (Au and Tp)")
    parser.add_argument("--masks", type=str, default="data/CASIAv2_raw/masks", help="Path to community ground truth masks")
    parser.add_argument("--output", type=str, default="data_pipeline/raw/CASIAv2", help="Output path for dataloader")
    args = parser.parse_args()
    
    format_casiav2(args.images, args.masks, args.output)
