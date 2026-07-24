import torch
import cv2
import numpy as np

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
    
    # Save Image 2
    img2 = probe_images[2]
    img2_np = (img2.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img2_np = cv2.cvtColor(img2_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite("offender_image_2.png", img2_np)
    
    # Save Image 10
    img10 = probe_images[10]
    img10_np = (img10.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img10_np = cv2.cvtColor(img10_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite("offender_image_10.png", img10_np)
    
    print("Saved offender_image_2.png and offender_image_10.png")
