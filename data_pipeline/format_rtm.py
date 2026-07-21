import os
import cv2
import glob
import numpy as np
from pathlib import Path
import argparse

def format_rtm(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    images_dir = input_path / 'images'
    masks_dir = input_path / 'masks'
    
    if not images_dir.exists():
        print(f"Error: Expected 'images' folder in {input_dir}")
        return
        
    # Find all images (png, tif, jpg, etc.)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(images_dir.glob(ext))
        image_files.extend(images_dir.glob(ext.upper()))
        
    print(f"Found {len(image_files)} images in {images_dir}. Processing...")
    
    processed = 0
    authentic = 0
    tampered = 0
    
    for img_path in image_files:
        stem = img_path.stem
        
        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: Could not read {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        # Save image strictly as .jpg in output directory
        out_img_path = out_path / f"{stem}.jpg"
        cv2.imwrite(str(out_img_path), img)
        
        # Look for corresponding mask in masks_dir
        mask_path = None
        if masks_dir.exists():
            for ext in ['.png', '.jpg', '.tif', '.tiff', '.jpeg']:
                possible_mask = masks_dir / f"{stem}{ext}"
                if possible_mask.exists():
                    mask_path = possible_mask
                    break
                
        out_mask_path = out_path / f"{stem}_mask.png"
        
        if mask_path:
            # Tampered image with mask
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                # Resize if necessary to perfectly match image
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(out_mask_path), mask)
            tampered += 1
        else:
            # Authentic image: Generate an all-zero (all-black) mask
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
    parser = argparse.ArgumentParser(description="Format RTM dataset for MVSS-Net Lite")
    parser.add_argument("--input", type=str, default="data/RTM_raw", help="Path to raw RTM dataset containing 'images' and 'masks' folders")
    parser.add_argument("--output", type=str, default="data_pipeline/raw/RTM", help="Output path for formatted dataset")
    args = parser.parse_args()
    
    print(f"Formatting RTM dataset from '{args.input}' to '{args.output}'...")
    format_rtm(args.input, args.output)
