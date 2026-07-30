import torch
import cv2
import numpy as np
import os

# Ensure deterministic splits so val set is genuinely held out from train set
torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)

from data_pipeline.dataset_loader import get_dataloader

def extract_probe_images():
    print("Extracting 20 held-out dense-text authentic images...")
    _, val_loader, _ = get_dataloader(['RTM', 'MIDV500'], batch_size=1, is_train=False, return_splits=True)
    
    probe_images = []
    for imgs, masks, edges in val_loader:
        if masks[0].sum() == 0:
            img_np = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.Canny(img_gray, 100, 200)
            edge_density = np.sum(img_edges > 0) / img_edges.size
            
            if edge_density > 0.03:
                probe_images.append(imgs[0])
            
            if len(probe_images) == 20:
                break
                
    return probe_images

if __name__ == "__main__":
    probe_images = extract_probe_images()
    
    os.makedirs("successful_probe_images", exist_ok=True)
    
    for i, img in enumerate(probe_images):
        if i in [2, 10]:
            continue # Already saved offenders
            
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"successful_probe_images/probe_image_{i}.png", img_np)
        
    print("Saved the other 18 images to successful_probe_images/")
