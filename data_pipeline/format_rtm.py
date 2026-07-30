import os
import cv2
import glob
import numpy as np
from pathlib import Path
import argparse

def format_rtm(images_dir: str, masks_dir: str, output_dir: str):
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not images_path.exists():
        print(f"Error: Expected images folder at {images_dir}")
        return
        
    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(images_path.glob(ext))
        image_files.extend(images_path.glob(ext.upper()))
        
    print(f"Found {len(image_files)} images in {images_dir}. Processing...")
    
    processed = 0
    authentic = 0
    tampered = 0
    
    for img_path in image_files:
        stem = img_path.stem
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        out_img_path = out_path / f"{stem}.jpg"
        cv2.imwrite(str(out_img_path), img)
        
        mask_path = None
        if masks_path.exists():
            for ext in ['.png', '.jpg', '.tif', '.tiff', '.jpeg']:
                possible_mask = masks_path / f"{stem}{ext}"
                if possible_mask.exists():
                    mask_path = possible_mask
                    break
                
        out_mask_path = out_path / f"{stem}_mask.png"
        
        if mask_path:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_mask_path), mask)
            tampered += 1
        else:
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.imwrite(str(out_mask_path), mask)
            authentic += 1
            
        processed += 1
        if processed % 500 == 0:
            print(f"Processed {processed} images...")

    print(f"\\nDone! Total Processed: {processed}")
    print(f"Tampered (mask found): {tampered}")
    print(f"Authentic (zero-mask generated): {authentic}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format RTM dataset")
    parser.add_argument("--images", type=str, default="data/RealTextManipulation/JPEGImages", help="Path to RTM images")
    parser.add_argument("--masks", type=str, default="data/RealTextManipulation/SegmentationClass", help="Path to RTM masks")
    parser.add_argument("--output", type=str, default="data_pipeline/raw/RTM", help="Output path for dataloader")
    args = parser.parse_args()
    
    format_rtm(args.images, args.masks, args.output)
