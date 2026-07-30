import torch
import cv2
import numpy as np
import os

torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)

from model.network import MVSSNetLite
from run_domain_split_pilot import extract_split_probe_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_heatmap_overlay(img_tensor, prob_map, filepath):
    # img_tensor: [3, 256, 256] in [0, 1]
    # prob_map: [256, 256] in [0, 1]
    
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Create a red heatmap
    heatmap = np.zeros_like(img_np)
    heatmap[:, :, 2] = (prob_map * 255).astype(np.uint8) # Red channel
    
    # Overlay (50% image, 50% heatmap)
    overlay = cv2.addWeighted(img_np, 0.7, heatmap, 0.5, 0)
    
    cv2.imwrite(filepath, overlay)

if __name__ == '__main__':
    rtm_tensor, _ = extract_split_probe_images()
    model = MVSSNetLite().to(device)
    model.eval()
    
    os.makedirs('peak_offenders', exist_ok=True)
    
    indices = [10, 14, 19, 16] # 10, 14, 19 are peaks. 16 is the non-hallucinating one.
    
    with torch.no_grad():
        for idx in indices:
            img = rtm_tensor[idx].unsqueeze(0)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            
            save_heatmap_overlay(rtm_tensor[idx], prob, f"peak_offenders/heatmap_untrained_{idx}.png")
            
            # Also save the raw image for 16 since we haven't seen it yet
            if idx == 16:
                img_np = (rtm_tensor[idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"peak_offenders/offender_image_{idx}.png", img_np)
                
    print("Saved heatmaps for 10, 14, 19, and 16.")
