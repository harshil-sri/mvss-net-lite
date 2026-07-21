import os
import cv2
import numpy as np
from pathlib import Path
import argparse

def format_midv500(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning MIDV-500 dataset in {input_dir}...")
    
    # MIDV-500 contains mostly authentic ID images/frames. 
    # We will treat all of them as authentic (all-zero mask).
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(input_path.rglob(ext))
        image_files.extend(input_path.rglob(ext.upper()))
        
    print(f"Found {len(image_files)} frames/images in MIDV-500.")
    
    processed = 0
    
    for img_path in image_files:
        stem = f"midv500_{img_path.parent.name}_{img_path.stem}"
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        out_img_path = out_path / f"{stem}.jpg"
        out_mask_path = out_path / f"{stem}_mask.png"
        
        # Save image
        cv2.imwrite(str(out_img_path), img)
        
        # Generate authentic all-zero mask
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(out_mask_path), mask)
        
        processed += 1
        if processed % 1000 == 0:
            print(f"Processed {processed} MIDV-500 images...")

    print(f"\\nMIDV-500 Formatting Done! Total authentic images processed: {processed}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format MIDV-500 dataset")
    parser.add_argument("--input", type=str, default="data/MIDV500_raw", help="Path to raw MIDV-500 dataset")
    parser.add_argument("--output", type=str, default="data_pipeline/raw/MIDV500", help="Output path for dataloader")
    args = parser.parse_args()
    
    format_midv500(args.input, args.output)
