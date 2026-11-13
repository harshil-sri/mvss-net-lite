import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def format_defacto(images_dir: str, masks_dir: str, output_dir: str):
    images_path = Path(images_dir)
    masks_path = Path(masks_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning DEFACTO dataset images in {images_dir}...")
    
    processed = 0
    
    for img_file in images_path.rglob('*'):
        if not img_file.is_file() or img_file.suffix not in ['.tif', '.png', '.jpg', '.jpeg']:
            continue
            
        stem = img_file.stem
        
        # Find corresponding mask
        mask_file = None
        for ext in ['.tif', '.png', '.jpg', '.jpeg']:
            possible_mask = masks_path / f"{stem}{ext}"
            if possible_mask.exists():
                mask_file = possible_mask
                break
                
        if mask_file is None:
            continue
            
        # Load and save Image
        img = cv2.imread(str(img_file))
        if img is None: continue
        h, w = img.shape[:2]
        
        # Load and save Mask
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None: continue
        
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            
        out_img_path = out_path / f"defacto_{stem}.jpg"
        out_mask_path = out_path / f"defacto_{stem}_mask.png"
        
        cv2.imwrite(str(out_img_path), img)
        cv2.imwrite(str(out_mask_path), mask)
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed} DEFACTO pairs...")

    print(f"\\nDEFACTO Formatting Done! Total pairs processed: {processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format DEFACTO dataset")
    parser.add_argument("--images", type=str, default="data/copymove_img/img", help="Path to raw DEFACTO images")
    parser.add_argument("--masks", type=str, default="data/copymove_annotations/probe_mask", help="Path to raw DEFACTO masks")
    parser.add_argument("--output", type=str, default="data_pipeline/raw/DEFACTO", help="Output path for dataloader")
    args = parser.parse_args()
    
    format_defacto(args.images, args.masks, args.output)
