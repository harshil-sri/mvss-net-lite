import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure deterministic splits so val set is genuinely held out from train set
torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)

from model.network import MVSSNetLite
from model.train import CombinedLoss
from data_pipeline.dataset_loader import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
LOG_FILE = "reports/pilot_results.txt"

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def extract_probe_images():
    log_and_print("Extracting 20 held-out dense-text authentic images...")
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
                
    probe_tensor = torch.stack(probe_images).to(device)
    log_and_print(f"Extraction complete. Found {probe_tensor.size(0)} images.")
    return probe_tensor

def run_probe(model, probe_tensor, step_name):
    model.eval()
    max_probs = []
    hallucinated_pixels = []
    with torch.no_grad():
        for i in range(probe_tensor.size(0)):
            img = probe_tensor[i].unsqueeze(0)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            max_probs.append(prob.max())
            hallucinated_pixels.append(np.sum(prob > 0.5))
            
    log_and_print(f"\n--- PROBE AT {step_name} ---")
    log_and_print(f"Avg Max Prob: {np.mean(max_probs):.4f} (Peak: {np.max(max_probs):.4f})")
    log_and_print(f"Avg Hallucinated Pixels (>0.5): {np.mean(hallucinated_pixels):.2f}")
    log_and_print(f"Total Hallucinating Images (out of {probe_tensor.size(0)}): {sum(1 for x in hallucinated_pixels if x > 0)}")
    log_and_print("---------------------------\n")

def run_stage1():
    log_and_print("\n" + "="*50)
    log_and_print("STAGE 1 PILOT: CASIAv2 + DEFACTO (1 Epoch)")
    log_and_print("="*50)
    
    train_loader, _, _ = get_dataloader(['CASIAv2', 'DEFACTO'], batch_size=BATCH_SIZE, is_train=True, return_splits=True)
    
    model = MVSSNetLite().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    seg_criterion = CombinedLoss(pos_weight_val=39.49)
    edge_criterion = CombinedLoss(pos_weight_val=817.24, use_tversky=True)
    
    model.train()
    history = []
    
    for batch_idx, (imgs, masks, edges) in enumerate(train_loader):
        imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
        optimizer.zero_grad()
        
        pred_seg, pred_edge = model(imgs)
        loss_seg = seg_criterion(pred_seg, masks)
        loss_edge = edge_criterion(pred_edge, edges)
        loss_total = loss_seg + loss_edge
        
        loss_total.backward()
        optimizer.step()
        model.backbone.noise_extractor.normalize_weights()
        
        history.append(loss_total.item())
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss_total.item():.4f}")
            
    plt.figure()
    plt.plot(history)
    plt.title("Stage 1 Pilot Loss Curve")
    plt.savefig("reports/pilot_stage1_loss.png")
    log_and_print("Stage 1 Pilot complete. Loss curve saved to reports/pilot_stage1_loss.png")

def run_stage2_and_ablation(probe_tensor):
    log_and_print("\n" + "="*50)
    log_and_print("STAGE 2 PILOT: RTM + MIDV500 (2 Epochs, Fresh Init)")
    log_and_print("="*50)
    
    train_loader, val_loader, _ = get_dataloader(['RTM', 'MIDV500'], batch_size=BATCH_SIZE, is_train=True, return_splits=True)
    total_batches = len(train_loader)
    
    model = MVSSNetLite().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    seg_criterion = CombinedLoss(pos_weight_val=230.63)
    edge_criterion = CombinedLoss(pos_weight_val=7766.23, use_tversky=True)
    
    run_probe(model, probe_tensor, "0% (Untrained)")
    
    for epoch in range(1, 3):
        model.train()
        for batch_idx, (imgs, masks, edges) in enumerate(train_loader):
            imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
            optimizer.zero_grad()
            
            pred_seg, pred_edge = model(imgs)
            loss_seg = seg_criterion(pred_seg, masks)
            loss_edge = edge_criterion(pred_edge, edges)
            loss_total = loss_seg + loss_edge
            
            loss_total.backward()
            optimizer.step()
            model.backbone.noise_extractor.normalize_weights()
            
            if batch_idx == total_batches // 2:
                run_probe(model, probe_tensor, f"Epoch {epoch} - 50% through epoch")
                model.train()
                
        run_probe(model, probe_tensor, f"Epoch {epoch} - 100% complete")
        
    log_and_print("\n" + "="*50)
    log_and_print("BAYAR NOISE BRANCH ABLATION TEST")
    log_and_print("="*50)
    
    ablation_batches = []
    for imgs, masks, edges in val_loader:
        if masks.sum() > 0:
            ablation_batches.append((imgs.to(device), masks.to(device), edges.to(device)))
        if len(ablation_batches) == 5:
            break
            
    # Intact evaluation
    model.eval()
    intact_losses = []
    with torch.no_grad():
        for imgs, masks, edges in ablation_batches:
            p_seg, p_edge = model(imgs)
            intact_losses.append((seg_criterion(p_seg, masks) + edge_criterion(p_edge, edges)).item())
            
    # Zero out the noise branch safely
    with torch.no_grad():
        model.backbone.noise_extractor.constrained_conv.weight.data.zero_()
        
    zeroed_losses = []
    with torch.no_grad():
        for imgs, masks, edges in ablation_batches:
            p_seg, p_edge = model(imgs)
            zeroed_losses.append((seg_criterion(p_seg, masks) + edge_criterion(p_edge, edges)).item())
            
    log_and_print(f"Average Loss (Intact BayarConv): {np.mean(intact_losses):.4f}")
    log_and_print(f"Average Loss (Zeroed BayarConv): {np.mean(zeroed_losses):.4f}")
    
    diff = np.mean(zeroed_losses) - np.mean(intact_losses)
    log_and_print(f"\nDifference: {diff:.4f}")
    if diff > 0.1:
        log_and_print("CONCLUSION: Noise branch is contributing significantly to the predictions!")
    elif diff < -0.1:
        log_and_print("CONCLUSION: Noise branch is actively harming predictions (needs fixing).")
    else:
        log_and_print("CONCLUSION: Noise branch has negligible impact. Augmentations likely destroyed the signal.")

if __name__ == '__main__':
    os.makedirs("reports", exist_ok=True)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    probe_tensor = extract_probe_images()
    run_stage1()
    run_stage2_and_ablation(probe_tensor)
