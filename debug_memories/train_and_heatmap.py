import torch
import cv2
import numpy as np
import os

torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)

from model.network import MVSSNetLite
from model.train import CombinedLoss
from data_pipeline.dataset_loader import get_dataloader
from run_domain_split_pilot import extract_split_probe_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_heatmap_overlay(img_tensor, prob_map, filepath):
    img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    heatmap = np.zeros_like(img_np)
    heatmap[:, :, 2] = (prob_map * 255).astype(np.uint8) # Red channel
    overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.6, 0)
    cv2.imwrite(filepath, overlay)

def train_and_generate_heatmaps():
    print("Extracting probe images...")
    rtm_tensor, _ = extract_split_probe_images()
    
    print("Initializing trained model run (1 Epoch)...")
    train_loader, _, _ = get_dataloader(['RTM', 'MIDV500'], batch_size=8, is_train=True, return_splits=True, use_balanced_sampler=True)
    
    model = MVSSNetLite().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    seg_criterion = CombinedLoss(pos_weight_val=230.63)
    edge_criterion = CombinedLoss(pos_weight_val=6216.84, use_tversky=True)
    
    model.train()
    for batch_idx, (imgs, masks, edges) in enumerate(train_loader):
        imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
        optimizer.zero_grad()
        pred_seg, pred_edge = model(imgs)
        loss_total = seg_criterion(pred_seg, masks) + edge_criterion(pred_edge, edges)
        loss_total.backward()
        optimizer.step()
        model.backbone.noise_extractor.normalize_weights()
        
    print("Training complete. Generating trained heatmaps for peak images...")
    
    model.eval()
    os.makedirs('peak_offenders', exist_ok=True)
    indices = [10, 14, 19, 16]
    
    with torch.no_grad():
        for idx in indices:
            img = rtm_tensor[idx].unsqueeze(0)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            
            save_heatmap_overlay(rtm_tensor[idx], prob, f"peak_offenders/heatmap_trained_{idx}.png")
            
    print("Trained heatmaps generated and saved.")

if __name__ == '__main__':
    train_and_generate_heatmaps()
