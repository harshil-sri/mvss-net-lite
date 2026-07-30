import torch
import cv2
import numpy as np
import os
import random

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_FILE = "reports/domain_split_stage1_results.txt"

def log_and_print(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def run_probe_for_epoch(epoch):
    chkpt_path = f"model/checkpoints/stage1_mvss_lite_ep{epoch}.pt"
    if not os.path.exists(chkpt_path):
        log_and_print(f"Checkpoint {chkpt_path} not found!")
        return

    model = MVSSNetLite().to(device)
    chkpt = torch.load(chkpt_path, map_location=device)
    if 'model_state_dict' in chkpt:
        model.load_state_dict(chkpt['model_state_dict'])
    elif 'model' in chkpt:
        model.load_state_dict(chkpt['model'])
    else:
        model.load_state_dict(chkpt)
    model.eval()
    
    log_and_print(f"\n{'='*50}\nEVALUATING EPOCH {epoch}\n{'='*50}")

    def evaluate_set(tensor, domain_name):
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
        
        log_and_print(f"\n--- FP PROBE ({domain_name}) ---")
        log_and_print(f"Avg Max Prob: {np.mean(max_probs):.4f} (Peak: {peak_val:.4f} on Image #{peak_idx})")
        
        for t in thresholds:
            avg_pixels = np.mean(hallucinated_pixels[t])
            count = len(hallucinating_images[t])
            log_and_print(f"Threshold {t}: {count}/{tensor.size(0)} hallucinating (Avg Pixels: {avg_pixels:.2f})")
        log_and_print("---------------------------\n")

    # Evaluate on the loaded probe sets
    evaluate_set(casia_auth, "CASIAv2 Authentic (Should be 0 FP)")
    evaluate_set(casia_forged, "CASIAv2 Forged (Should be High TP)")
    evaluate_set(defacto_forged, "DEFACTO Forged (Should be High TP)")

if __name__ == '__main__':
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
        
    log_and_print("Building probe sets directly from the new unified manifest...")
    
    # We can just manually read the manifest to grab 50 of each
    import json
    with open('reports/manifest.json', 'r') as f:
        manifest = json.load(f)
        
    val_set = manifest['val']
    
    casia_auth_paths = []
    casia_forged_paths = []
    defacto_forged_paths = []
    
    for s in val_set:
        img, mask, dataset = s['image'], s['mask'], s['dataset']
        is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
        
        if dataset == 'CASIAv2':
            if is_forged and len(casia_forged_paths) < 50:
                casia_forged_paths.append(s)
            elif not is_forged and len(casia_auth_paths) < 50:
                casia_auth_paths.append(s)
        elif dataset == 'DEFACTO':
            if is_forged and len(defacto_forged_paths) < 50:
                defacto_forged_paths.append(s)
                
    # Now we use ForgeryDataset to process them properly
    from data_pipeline.dataset_loader import ForgeryDataset
    casia_auth_ds = ForgeryDataset([(p['image'], p['mask']) for p in casia_auth_paths], is_train=False)
    casia_forged_ds = ForgeryDataset([(p['image'], p['mask']) for p in casia_forged_paths], is_train=False)
    defacto_forged_ds = ForgeryDataset([(p['image'], p['mask']) for p in defacto_forged_paths], is_train=False)
    
    # Extract to tensors
    def ds_to_tensor(ds):
        imgs = []
        for i in range(len(ds)):
            imgs.append(ds[i][0])
        return torch.stack(imgs).to(device)
        
    casia_auth = ds_to_tensor(casia_auth_ds)
    casia_forged = ds_to_tensor(casia_forged_ds)
    defacto_forged = ds_to_tensor(defacto_forged_ds)
    
    log_and_print(f"Extracted {casia_auth.size(0)} CASIAv2 Authentic")
    log_and_print(f"Extracted {casia_forged.size(0)} CASIAv2 Forged")
    log_and_print(f"Extracted {defacto_forged.size(0)} DEFACTO Forged")
    
    for ep in [30, 35, 40, 45, 50]:
        run_probe_for_epoch(ep)
