import torch
import cv2
import numpy as np
import os
import random

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from model.network import MVSSNetLite
from model.train import CombinedLoss
from data_pipeline.dataset_loader import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_FILE = "reports/stage2_proper_results.txt"

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def extract_split_probe_images():
    log_and_print("Loading specific domain-split probe images from indices...")
    _, rtm_val, _ = get_dataloader(['RTM'], batch_size=1, is_train=False, return_splits=True)
    
    with open('reports/probe_indices.txt', 'r') as f:
        lines = f.readlines()
        old_idx = [int(x) for x in lines[0].strip().replace('OLD:', '').split(',')]
        new_idx = [int(x) for x in lines[1].strip().replace('NEW:', '').split(',')]
        
    old_probes = []
    new_probes = []
    
    for i, (imgs, masks, edges) in enumerate(rtm_val):
        if i in old_idx:
            old_probes.append(imgs[0])
        if i in new_idx:
            new_probes.append(imgs[0])
            
        if len(old_probes) == len(old_idx) and len(new_probes) == len(new_idx):
            break
            
    log_and_print(f"Extraction complete. Found {len(old_probes)} OLD and {len(new_probes)} NEW probes.")
    return torch.stack(old_probes).to(device), torch.stack(new_probes).to(device)

def run_split_probe(model, tensor, domain_name, step_name):
    if tensor is None:
        return
        
    model.eval()
    max_probs = []
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    hallucinated_pixels = {t: [] for t in thresholds}
    hallucinating_images = {t: [] for t in thresholds}
    
    with torch.no_grad():
        for i in range(tensor.size(0)):
            img = tensor[i].unsqueeze(0)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            max_probs.append(prob.max())
            
            for t in thresholds:
                pixels = np.sum(prob > t)
                hallucinated_pixels[t].append(pixels)
                if pixels > 0:
                    hallucinating_images[t].append(i)
            
    peak_val = np.max(max_probs)
    peak_idx = np.argmax(max_probs)
    
    log_and_print(f"\n--- FP PROBE AT {step_name} ({domain_name}) ---")
    log_and_print(f"Avg Max Prob: {np.mean(max_probs):.4f} (Peak: {peak_val:.4f} on Image #{peak_idx})")
    
    for t in thresholds:
        avg_pixels = np.mean(hallucinated_pixels[t])
        count = len(hallucinating_images[t])
        log_and_print(f"Threshold {t}: {count}/{tensor.size(0)} hallucinating (Avg Pixels: {avg_pixels:.2f})")
        if count > 0 and t == 0.5:
             log_and_print(f"Hallucinating Image Indices (>0.5): {hallucinating_images[t]}")
             
    log_and_print("---------------------------\n")

def run_stage2_proper():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    old_tensor, new_tensor = extract_split_probe_images()
    
    log_and_print("\n" + "="*50)
    log_and_print("STAGE 2 PROPER EXTENSION: Epoch 1 to 15 (Cosine Annealing)")
    log_and_print("="*50)
    
    train_loader, _, _ = get_dataloader(['RTM', 'MIDV500'], batch_size=8, is_train=True, return_splits=True, use_balanced_sampler=True)
    
    model = MVSSNetLite().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    
    seg_criterion = CombinedLoss(pos_weight_val=230.63)
    edge_criterion = CombinedLoss(pos_weight_val=1058.38, use_tversky=True)
    
    os.makedirs('model/checkpoints', exist_ok=True)
    
    for epoch in range(1, 16):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        log_and_print(f"\n--- Epoch {epoch}/15 (LR: {current_lr:.6f}) ---")
        
        for batch_idx, (imgs, masks, edges) in enumerate(train_loader):
            imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
            optimizer.zero_grad()
            
            pred_seg, pred_edge = model(imgs)
            loss_total = seg_criterion(pred_seg, masks) + edge_criterion(pred_edge, edges)
            
            loss_total.backward()
            optimizer.step()
            model.backbone.noise_extractor.normalize_weights()
            
        scheduler.step()
                
        if epoch in [5, 8, 11, 15]:
            chkpt_path = f"model/checkpoints/stage2_proper_ep{epoch}.pt"
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'lr': current_lr
            }, chkpt_path)
            log_and_print(f"Checkpoint saved to {chkpt_path}")
            
            run_split_probe(model, old_tensor, "RTM Authentic Tabular (OLD PROBE SET)", f"Epoch {epoch}")
            run_split_probe(model, new_tensor, "RTM Authentic Tabular (NEW UNSEEN SET)", f"Epoch {epoch}")
            
            os.system(f"python check_recall.py {chkpt_path} >> {LOG_FILE}")
            
if __name__ == '__main__':
    run_stage2_proper()
