import torch
import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)

from model.network import MVSSNetLite
from model.train import CombinedLoss
from data_pipeline.dataset_loader import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_FILE = "reports/domain_split_pilot_results.txt"

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def extract_split_probe_images():
    log_and_print("Extracting domain-split probe images (RTM vs MIDV500)...")
    
    # We load val loaders separately for each dataset to guarantee domain
    _, rtm_val, _ = get_dataloader(['RTM'], batch_size=1, is_train=False, return_splits=True)
    _, midv_val, _ = get_dataloader(['MIDV500'], batch_size=1, is_train=False, return_splits=True)
    
    rtm_probes = []
    # RTM doesn't have true authentic (untampered) images in the split according to our check.
    # But wait, earlier we found 20 images with 0 GT edge pixels. 
    # Let's extract any image that has 0 forged pixels in the mask, regardless of what dataset it's from,
    # but track which dataloader it came from.
    
    for imgs, masks, edges in rtm_val:
        if masks[0].sum() == 0:
            img_np = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.Canny(img_gray, 100, 200)
            edge_density = np.sum(img_edges > 0) / img_edges.size
            if edge_density > 0.03:
                rtm_probes.append(imgs[0])
            if len(rtm_probes) == 20:
                break
                
    midv_probes = []
    for imgs, masks, edges in midv_val:
        if masks[0].sum() == 0:
            img_np = (imgs[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            img_edges = cv2.Canny(img_gray, 100, 200)
            edge_density = np.sum(img_edges > 0) / img_edges.size
            if edge_density > 0.03:
                midv_probes.append(imgs[0])
            if len(midv_probes) == 20:
                break
                
    log_and_print(f"Extraction complete. Found {len(rtm_probes)} RTM authentics and {len(midv_probes)} MIDV500 authentics.")
    
    rtm_tensor = torch.stack(rtm_probes).to(device) if rtm_probes else None
    midv_tensor = torch.stack(midv_probes).to(device) if midv_probes else None
    
    return rtm_tensor, midv_tensor

def run_split_probe(model, tensor, domain_name, step_name):
    if tensor is None:
        log_and_print(f"\n--- PROBE AT {step_name} ({domain_name}) ---")
        log_and_print("No images found for this domain.")
        return
        
    model.eval()
    max_probs = []
    hallucinated_pixels = []
    with torch.no_grad():
        for i in range(tensor.size(0)):
            img = tensor[i].unsqueeze(0)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            max_probs.append(prob.max())
            hallucinated_pixels.append(np.sum(prob > 0.5))
            
    peak_val = np.max(max_probs)
    peak_idx = np.argmax(max_probs)
    hallucinating_indices = [idx for idx, pixels in enumerate(hallucinated_pixels) if pixels > 0]
    
    log_and_print(f"\n--- PROBE AT {step_name} ({domain_name}) ---")
    log_and_print(f"Avg Max Prob: {np.mean(max_probs):.4f} (Peak: {peak_val:.4f} on Image #{peak_idx})")
    log_and_print(f"Avg Hallucinated Pixels (>0.5): {np.mean(hallucinated_pixels):.2f}")
    log_and_print(f"Total Hallucinating Images: {len(hallucinating_indices)}/{tensor.size(0)}")
    if hallucinating_indices:
        log_and_print(f"Hallucinating Image Indices: {hallucinating_indices}")
    log_and_print("---------------------------\n")

def run_stage2_split_pilot(rtm_tensor, midv_tensor):
    log_and_print("\n" + "="*50)
    log_and_print("STAGE 2 PILOT: RTM + MIDV500 (2 Epochs, Balanced Sampler)")
    log_and_print("="*50)
    
    train_loader, _, _ = get_dataloader(['RTM', 'MIDV500'], batch_size=8, is_train=True, return_splits=True, use_balanced_sampler=True)
    total_batches = len(train_loader)
    
    model = MVSSNetLite().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    seg_criterion = CombinedLoss(pos_weight_val=230.63)
    edge_criterion = CombinedLoss(pos_weight_val=6216.84, use_tversky=True)
    
    run_split_probe(model, rtm_tensor, "RTM", "0% (Untrained)")
    run_split_probe(model, midv_tensor, "MIDV500", "0% (Untrained)")
    
    for epoch in range(1, 3):
        model.train()
        for batch_idx, (imgs, masks, edges) in enumerate(train_loader):
            imgs, masks, edges = imgs.to(device), masks.to(device), edges.to(device)
            optimizer.zero_grad()
            
            pred_seg, pred_edge = model(imgs)
            loss_total = seg_criterion(pred_seg, masks) + edge_criterion(pred_edge, edges)
            
            loss_total.backward()
            optimizer.step()
            model.backbone.noise_extractor.normalize_weights()
            
            if batch_idx == total_batches // 2:
                run_split_probe(model, rtm_tensor, "RTM", f"Epoch {epoch} - 50%")
                run_split_probe(model, midv_tensor, "MIDV500", f"Epoch {epoch} - 50%")
                model.train()
                
        run_split_probe(model, rtm_tensor, "RTM", f"Epoch {epoch} - 100%")
        run_split_probe(model, midv_tensor, "MIDV500", f"Epoch {epoch} - 100%")

    return model

def run_split_ablation(model, dataset_name):
    log_and_print(f"\n================ {dataset_name} ABLATION ================")
    _, val_loader, _ = get_dataloader([dataset_name], batch_size=8, is_train=False, return_splits=True)
    
    seg_criterion = CombinedLoss(pos_weight_val=230.63)
    edge_criterion = CombinedLoss(pos_weight_val=6216.84, use_tversky=True)
    
    ablation_batches = []
    for imgs, masks, edges in val_loader:
        if masks.sum() > 0:
            ablation_batches.append((imgs.to(device), masks.to(device), edges.to(device)))
        if len(ablation_batches) == 5:
            break
            
    if not ablation_batches:
        log_and_print(f"No forged batches found for {dataset_name}.")
        return
            
    # Intact evaluation
    model.eval()
    intact_losses = []
    with torch.no_grad():
        for imgs, masks, edges in ablation_batches:
            p_seg, p_edge = model(imgs)
            intact_losses.append((seg_criterion(p_seg, masks) + edge_criterion(p_edge, edges)).item())
            
    # Zero out the noise branch
    with torch.no_grad():
        original_weights = model.backbone.noise_extractor.constrained_conv.weight.data.clone()
        model.backbone.noise_extractor.constrained_conv.weight.data.zero_()
        
    zeroed_losses = []
    with torch.no_grad():
        for imgs, masks, edges in ablation_batches:
            p_seg, p_edge = model(imgs)
            zeroed_losses.append((seg_criterion(p_seg, masks) + edge_criterion(p_edge, edges)).item())
            
    # Restore weights so we can run the next ablation cleanly
    with torch.no_grad():
        model.backbone.noise_extractor.constrained_conv.weight.data.copy_(original_weights)
            
    log_and_print(f"Average Loss (Intact BayarConv): {np.mean(intact_losses):.4f}")
    log_and_print(f"Average Loss (Zeroed BayarConv): {np.mean(zeroed_losses):.4f}")
    
    diff = np.mean(zeroed_losses) - np.mean(intact_losses)
    log_and_print(f"Difference: {diff:.4f}")

if __name__ == '__main__':
    os.makedirs("reports", exist_ok=True)
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    rtm_tensor, midv_tensor = extract_split_probe_images()
    model = run_stage2_split_pilot(rtm_tensor, midv_tensor)
    
    log_and_print("\n" + "="*50)
    log_and_print("DOMAIN SPLIT BAYAR ABLATION TESTS")
    log_and_print("="*50)
    run_split_ablation(model, "RTM")
    run_split_ablation(model, "MIDV500")
