import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
import copy
from tqdm import tqdm

from model.network import MVSSNetLite
from model.train import CombinedLoss
from data_pipeline.dataset_loader import ForgeryDataset

def get_probe_set(manifest_path, n_auth=200, n_forged=200):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    rtm_test = [item for item in manifest['test'] if item.get('dataset') == 'RTM']
    
    auth_samples = []
    forged_samples = []
    
    for item in rtm_test:
        img_path = item['image']
        mask_path = item['mask']
        # simple heuristic: 'good' in filename means authentic, else forged
        # Or load mask and check
        if 'good' in img_path.lower():
            if len(auth_samples) < n_auth:
                auth_samples.append((img_path, mask_path))
        else:
            if len(forged_samples) < n_forged:
                forged_samples.append((img_path, mask_path))
                
        if len(auth_samples) == n_auth and len(forged_samples) == n_forged:
            break
            
    return auth_samples + forged_samples

def get_edges(masks):
    # masks: (B, 1, H, W)
    # Simple Sobel or morphological edge
    b, c, h, w = masks.shape
    edges = torch.zeros_like(masks)
    
    # Simple shift to find boundaries
    masks_shift_r = torch.roll(masks, 1, dims=3)
    masks_shift_l = torch.roll(masks, -1, dims=3)
    masks_shift_d = torch.roll(masks, 1, dims=2)
    masks_shift_u = torch.roll(masks, -1, dims=2)
    
    diff = (masks != masks_shift_r) | (masks != masks_shift_l) | \
           (masks != masks_shift_d) | (masks != masks_shift_u)
    
    edges[diff] = 1.0
    return edges

def train_model(model, dataloader, use_edge_loss, num_steps=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    seg_criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight_val=100.0)
    edge_criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight_val=100.0, use_tversky=True)
    
    steps_done = 0
    pbar = tqdm(total=num_steps, desc=f"Training (Edge Loss: {use_edge_loss})")
    
    while steps_done < num_steps:
        for batch_data in dataloader:
            if steps_done >= num_steps:
                break
                
            images = batch_data[0].to(device)
            masks = batch_data[1].to(device)
            edges = batch_data[2].to(device)
            
            optimizer.zero_grad()
            seg_pred, edge_pred = model(images)
            
            loss = seg_criterion(seg_pred, masks)
            if use_edge_loss:
                loss += edge_criterion(edge_pred, edges)
                
            loss.backward()
            optimizer.step()
            
            steps_done += 1
            pbar.update(1)
            
    pbar.close()
    return model

def evaluate_model(model, dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_seg_preds = []
    all_masks = []
    edge_activations = []
    
    image_scores = []
    image_labels = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Evaluating"):
            images = batch_data[0].to(device)
            masks = batch_data[1].to(device)
            
            seg_pred, edge_pred = model(images)
            
            seg_probs = torch.sigmoid(seg_pred).cpu().numpy()
            masks_np = masks.cpu().numpy()
            
            all_seg_preds.append(seg_probs.flatten())
            all_masks.append(masks_np.flatten())
            
            edge_probs = torch.sigmoid(edge_pred)
            edge_activations.append(edge_probs.mean().item())
            
            # For image level metrics
            for i in range(images.shape[0]):
                img_score = seg_probs[i].max()
                img_label = 1 if masks_np[i].sum() > 0 else 0
                image_scores.append(img_score)
                image_labels.append(img_label)
                
    # Calculate pixel AUC
    # Subsample for speed if needed, but let's try full
    all_seg_preds_cat = np.concatenate(all_seg_preds)
    all_masks_cat = np.concatenate(all_masks)
    
    # random subsample 1M pixels for AUC to save memory/time
    idx = np.random.choice(len(all_seg_preds_cat), 1000000, replace=False)
    pixel_auc = roc_auc_score(all_masks_cat[idx], all_seg_preds_cat[idx])
    
    # Calculate auth_fp_at_rec85 on image level
    fpr, tpr, thresholds = roc_curve(image_labels, image_scores)
    # Find threshold where tpr (recall) is >= 0.85
    idx = np.where(tpr >= 0.85)[0]
    if len(idx) > 0:
        auth_fp = fpr[idx[0]]
    else:
        auth_fp = 1.0 # fallback
        
    edge_mean_act = np.mean(edge_activations)
    
    return {
        "pixel_auc": float(pixel_auc),
        "auth_fp_at_rec85": float(auth_fp),
        "edge_mean_activation": float(edge_mean_act)
    }

def main():
    manifest_path = "reports/manifest.json"
    checkpoint_path = "model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt"
    
    print("Building probe set...")
    samples = get_probe_set(manifest_path)
    print(f"Got {len(samples)} samples.")
    
    dataset = ForgeryDataset(samples, crop_size=256, is_train=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)
    
    eval_dataset = ForgeryDataset(samples, crop_size=256, is_train=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    print("Loading base model...")
    base_model = MVSSNetLite()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    base_model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Training Model A (With Edge Loss)...")
    model_a = copy.deepcopy(base_model)
    model_a = train_model(model_a, dataloader, use_edge_loss=True, num_steps=100)
    
    print("Evaluating Model A...")
    res_a = evaluate_model(model_a, eval_dataloader)
    
    print("Training Model B (Without Edge Loss)...")
    model_b = copy.deepcopy(base_model)
    model_b = train_model(model_b, dataloader, use_edge_loss=False, num_steps=100)
    
    print("Evaluating Model B...")
    res_b = evaluate_model(model_b, eval_dataloader)
    
    delta = {
        "pixel_auc": res_a["pixel_auc"] - res_b["pixel_auc"],
        "auth_fp_at_rec85": res_a["auth_fp_at_rec85"] - res_b["auth_fp_at_rec85"]
    }
    
    final_results = {
        "with_edge": res_a,
        "without_edge": res_b,
        "delta": delta
    }
    
    print("\nResults:")
    print(json.dumps(final_results, indent=2))
    
    output_path = "reports/edge_head_ablation.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
