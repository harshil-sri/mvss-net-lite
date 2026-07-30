import os
import cv2
import numpy as np
import torch
from data_pipeline.dataset_loader import ForgeryDataset

def check_vanishing_masks():
    print("Initializing dataset...")
    # Load RTM dataset (training mode to enable augmentations and resizing)
    ds = ForgeryDataset('data_pipeline/raw', ['RTM'], is_train=True)
    
    total_forged = 0
    vanished_masks = 0
    
    print("Scanning RTM samples...")
    for idx in range(len(ds.samples)):
        img_path, mask_path = ds.samples[idx]
        
        # Check if the raw image is forged
        raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask.sum() == 0:
            continue # Skip authentic images
            
        total_forged += 1
        
        # Pass it through the dataloader pipeline (which resizes to 256x256)
        # Using a fixed seed for deterministic testing, but we can just run a few passes
        _, mask_tensor, edge_tensor = ds[idx]
        
        if edge_tensor.sum() == 0:
            vanished_masks += 1
            
        if total_forged % 1000 == 0:
            print(f"Scanned {total_forged} forged images... ({vanished_masks} vanished so far)")
            
    print("\n--- RESULTS ---")
    print(f"Total Raw Forged Images Scanned: {total_forged}")
    print(f"Images where the forgery VANISHED after pipeline (0 positive pixels): {vanished_masks}")
    if total_forged > 0:
        print(f"Vanishing Rate: {(vanished_masks / total_forged) * 100:.2f}%")

if __name__ == '__main__':
    check_vanishing_masks()
