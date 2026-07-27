import torch
import cv2
import numpy as np
import os
import random

# Fixed seed so we always grab the same images for comparison
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_recall(checkpoint_path):
    print(f"\nEvaluating Recall for {checkpoint_path}")
    
    model = MVSSNetLite().to(device)
    chkpt = torch.load(checkpoint_path, map_location=device)
    if 'model' in chkpt:
        model.load_state_dict(chkpt['model'])
    else:
        model.load_state_dict(chkpt)
    model.eval()
    
    # Grab 50 RTM Forged images (general, not necessarily tabular)
    _, rtm_val, _ = get_dataloader(['RTM'], batch_size=1, is_train=False, return_splits=True)
    
    probes = []
    edges = []
    for imgs, masks, edgs in rtm_val:
        if masks[0].sum() > 0:
            probes.append(imgs[0])
            edges.append(edgs[0])
        if len(probes) == 50:
            break
            
    probes_tensor = torch.stack(probes).to(device)
    edges_tensor = torch.stack(edges).to(device)
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    tp_pixels = {t: [] for t in thresholds}
    
    with torch.no_grad():
        for i in range(probes_tensor.size(0)):
            img = probes_tensor[i].unsqueeze(0)
            gt_edge = edges_tensor[i].unsqueeze(0)
            
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            gt = gt_edge.squeeze().cpu().numpy()
            
            for t in thresholds:
                pred_binary = (prob > t)
                tp = np.logical_and(pred_binary, gt > 0.5).sum()
                tp_pixels[t].append(tp)
                
    for t in thresholds:
        avg_tp = np.mean(tp_pixels[t])
        print(f"Threshold {t}: Avg TP Pixels: {avg_tp:.2f}")

import sys
if __name__ == '__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            evaluate_recall(path)
        else:
            print(f"{path} not found.")
    else:
        ep3_path = 'model/checkpoints/stage2_mvss_lite_ep3.pt'
        ep5_path = 'model/checkpoints/stage2_mvss_lite_ep5.pt'
        
        if os.path.exists(ep3_path):
            evaluate_recall(ep3_path)
        else:
            print(f"{ep3_path} not found yet.")
            
        if os.path.exists(ep5_path):
            evaluate_recall(ep5_path)
        else:
            print(f"{ep5_path} not found yet.")
