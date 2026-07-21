import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def format_defacto(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning DEFACTO dataset in {input_dir}...")
    
    # DEFACTO has categories like splicing, copymove, etc.
    # Inside each category, images are usually in 'probe' and masks in 'probe_mask'
    
    processed = 0
    
    # Find all probe_mask directories
    for mask_dir in input_path.rglob('probe_mask'):
        # The images are usually in a sibling directory called 'probe'
        img_dir = mask_dir.parent / 'probe'
        if not img_dir.exists():
            continue
            
        print(f"Processing paired folders: {img_dir} & {mask_dir}")
        
        for mask_file in mask_dir.glob('*'):
            if not mask_file.is_file() or mask_file.suffix not in ['.tif', '.png', '.jpg']:
                continue
                
            stem = mask_file.stem
            
            # Find corresponding image
            img_file = None
            for ext in ['.tif', '.png', '.jpg', '.jpeg']:
                possible_img = img_dir / f"{stem}{ext}"
                if possible_img.exists():
                    img_file = possible_img
                    break
                    
            if img_file is None:
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
    parser.add_argument("--input", type=str, default="data/DEFACTO_raw", help="Path to raw DEFACTO dataset")
    parser.add_argument("--output", type=str, default="data_pipeline/raw/DEFACTO", help="Output path for dataloader")
    args = parser.parse_args()
    
    format_defacto(args.input, args.output)
