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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_FILE = "reports/stage2_training_results.txt"

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def extract_split_probe_images():
    log_and_print("Extracting domain-split probe images (RTM Authentic Tabular & RTM Forged Tabular)...")
    _, rtm_val, _ = get_dataloader(['RTM'], batch_size=1, is_train=False, return_splits=True)
    
    auth_probes = []
    forg_probes = []
    forg_edges = []
    
    for imgs, masks, edges in rtm_val:
        img_np = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        img_edges = cv2.Canny(img_gray, 100, 200)
        edge_density = np.sum(img_edges > 0) / img_edges.size
        
        # We specifically want images with dense tabular/grid structure (high edge density)
        if edge_density > 0.03:
            if masks[0].sum() == 0 and len(auth_probes) < 20:
                auth_probes.append(imgs[0])
            elif masks[0].sum() > 0 and len(forg_probes) < 20:
                forg_probes.append(imgs[0])
                forg_edges.append(edges[0])
                
        if len(auth_probes) == 20 and len(forg_probes) == 20:
            break
                
    log_and_print(f"Extraction complete. Found {len(auth_probes)} RTM Authentic Tabular and {len(forg_probes)} RTM Forged Tabular.")
    auth_tensor = torch.stack(auth_probes).to(device) if auth_probes else None
    forg_tensor = torch.stack(forg_probes).to(device) if forg_probes else None
    forg_gt_tensor = torch.stack(forg_edges).to(device) if forg_edges else None
    return auth_tensor, forg_tensor, forg_gt_tensor

def run_tp_probe(model, img_tensor, edge_tensor, step_name):
    if img_tensor is None:
        return
        
    model.eval()
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    tp_pixels = {t: [] for t in thresholds}
    
    with torch.no_grad():
        for i in range(img_tensor.size(0)):
            img = img_tensor[i].unsqueeze(0)
            gt_edge = edge_tensor[i].unsqueeze(0)
            
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            gt = gt_edge.squeeze().cpu().numpy()
            
            for t in thresholds:
                pred_binary = (prob > t)
                tp = np.logical_and(pred_binary, gt > 0.5).sum()
                tp_pixels[t].append(tp)
    
    log_and_print(f"\n--- TP PROBE AT {step_name} (RTM FORGED TABULAR) ---")
    for t in thresholds:
        avg_tp = np.mean(tp_pixels[t])
        log_and_print(f"Threshold {t}: Avg TP Pixels: {avg_tp:.2f}")
    log_and_print("---------------------------\n")

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

def run_stage2_training():
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    auth_tensor, forg_tensor, forg_gt_tensor = extract_split_probe_images()
    
    log_and_print("\n" + "="*50)
    log_and_print("STAGE 2 TRAINING: RTM + MIDV500 (5 Epochs, Balanced Sampler, Mask Fix)")
    log_and_print("="*50)
    
    train_loader, _, _ = get_dataloader(['RTM', 'MIDV500'], batch_size=8, is_train=True, return_splits=True, use_balanced_sampler=True)
    
    model = MVSSNetLite().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    seg_criterion = CombinedLoss(pos_weight_val=230.63)
    edge_criterion = CombinedLoss(pos_weight_val=1058.38, use_tversky=True)
    
    os.makedirs('model/checkpoints', exist_ok=True)
    
    run_split_probe(model, auth_tensor, "RTM Authentic Tabular", "0% (Untrained)")
    run_tp_probe(model, forg_tensor, forg_gt_tensor, "0% (Untrained)")

    
    for epoch in range(1, 6):
        model.train()
        for batch_idx, (imgs, masks, edges) in enumerate(train_loader):
            imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
            optimizer.zero_grad()
            
            pred_seg, pred_edge = model(imgs)
            loss_total = seg_criterion(pred_seg, masks) + edge_criterion(pred_edge, edges)
            
            loss_total.backward()
            optimizer.step()
            model.backbone.noise_extractor.normalize_weights()
                
        if epoch in [3, 5]:
            chkpt_path = f"model/checkpoints/stage2_mvss_lite_ep{epoch}.pt"
            torch.save(model.state_dict(), chkpt_path)
            log_and_print(f"Checkpoint saved to {chkpt_path}")
            
            run_split_probe(model, auth_tensor, "RTM Authentic Tabular", f"Epoch {epoch} - 100%")
            run_tp_probe(model, forg_tensor, forg_gt_tensor, f"Epoch {epoch} - 100%")

if __name__ == '__main__':
    run_stage2_training()
