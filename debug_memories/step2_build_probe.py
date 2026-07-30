import torch
import cv2
import numpy as np
import os
import random

# Fixed seed
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from data_pipeline.dataset_loader import get_dataloader

def build_new_probe():
    print("Extracting domain-split probe images (RTM Authentic Tabular)...")
    val_ds_loader = get_dataloader(['RTM'], batch_size=1, is_train=False, return_splits=True)[1]
    
    auth_tabular_old = []
    auth_tabular_new = []
    
    # Needs to match exactly the logic in run_stage2_training.py
    for i, (imgs, masks, edges) in enumerate(val_ds_loader):
        is_forged = masks[0].sum() > 0
        img_np = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(canny > 0) / canny.size
        
        is_table = edge_density > 0.03
        
        if not is_forged and is_table:
            if len(auth_tabular_old) < 20:
                auth_tabular_old.append(i)
            elif len(auth_tabular_new) < 20:
                auth_tabular_new.append(i)
                
        if len(auth_tabular_new) == 20:
            break
            
    print(f"Old probe set global indices: {auth_tabular_old}")
    print(f"New unseen probe set global indices: {auth_tabular_new}")
    
    overlap = set(auth_tabular_old).intersection(set(auth_tabular_new))
    if len(overlap) == 0:
        print("\nConfirmed: Zero overlap between old and new probe sets.")
    else:
        print(f"\nERROR: Overlap found: {overlap}")
        
    # Save the indices for Step 3
    with open('reports/probe_indices.txt', 'w') as f:
        f.write(f"OLD:{','.join(map(str, auth_tabular_old))}\n")
        f.write(f"NEW:{','.join(map(str, auth_tabular_new))}\n")
    print("Saved probe indices to reports/probe_indices.txt")

if __name__ == '__main__':
    build_new_probe()
