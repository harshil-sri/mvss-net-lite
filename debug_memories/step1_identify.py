import torch
import cv2
import numpy as np
import os
import random

# Fixed seed
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def identify_stubborn():
    print("Extracting domain-split probe images (RTM Authentic Tabular)...")
    _, val_loader, _ = get_dataloader(['RTM'], batch_size=1, is_train=False, return_splits=True)
    
    auth_tabular = []
    
    # Needs to match exactly the logic in run_stage2_training.py
    for i, (imgs, masks, edges) in enumerate(val_loader):
        is_forged = masks[0].sum() > 0
        img_np = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(canny > 0) / canny.size
        
        is_table = edge_density > 0.03
        
        if not is_forged and is_table:
            auth_tabular.append((i, imgs[0]))
            
        if len(auth_tabular) == 20:
            break
            
    # Load epoch 5 model
    model = MVSSNetLite().to(device)
    model.load_state_dict(torch.load('model/checkpoints/stage2_mvss_lite_ep5.pt', map_location=device))
    model.eval()
    
    print("\n--- Identifying >0.9 Hallucinating Images ---")
    hallucinating_indices = []
    
    with torch.no_grad():
        for local_idx, (global_idx, img) in enumerate(auth_tabular):
            img_t = img.unsqueeze(0).to(device)
            _, pred_edge = model(img_t)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            
            # > 0.9 threshold
            pred_binary = (prob > 0.9)
            if pred_binary.sum() > 10: # threshold used in sweep
                hallucinating_indices.append((local_idx, global_idx))
                
    print(f"Indices in the 20-image probe set that hallucinate >0.9: {[x[0] for x in hallucinating_indices]}")
    print(f"Global dataset indices of these images: {[x[1] for x in hallucinating_indices]}")
    
    # Check old pilot results if available
    old_log = 'reports/pilot_results.txt'
    if os.path.exists(old_log):
        with open(old_log, 'r') as f:
            print("\nFound old pilot results. Scanning for peak offenders...")
            for line in f:
                if "Peak" in line or "offender" in line.lower():
                    print(line.strip())
                    
    old_domain = 'debug_memories/domain_split_results.md'
    if os.path.exists(old_domain):
        print("\nChecking domain split results...")
        with open(old_domain, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "Peak" in line or "Indices" in line:
                    print(line.strip())

if __name__ == '__main__':
    identify_stubborn()
