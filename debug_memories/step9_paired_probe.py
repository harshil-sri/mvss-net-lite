import torch
import cv2
import numpy as np
import os
import json
from tqdm import tqdm
import time

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

# Enforce CPU to not disrupt Stage 2
device = torch.device('cpu')
print("Using device:", device)

def get_probe_sets(manifest_path, max_per_class=200):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    val_set = manifest['val']
    
    casia_auth_paths = []
    casia_forged_paths = []
    defacto_forged_paths = []
    
    for s in val_set:
        img, mask, dataset = s['image'], s['mask'], s['dataset']
        # For DEFACTO, some paths might be missing or empty? We assume they are correct.
        # But wait, determining if it's forged:
        if dataset == 'CASIAv2':
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if is_forged and len(casia_forged_paths) < max_per_class:
                casia_forged_paths.append(s)
            elif not is_forged and len(casia_auth_paths) < max_per_class:
                casia_auth_paths.append(s)
        elif dataset == 'DEFACTO':
            # DEFACTO are all forged in our datasets usually, let's verify mask
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if is_forged and len(defacto_forged_paths) < max_per_class:
                defacto_forged_paths.append(s)
                
    return casia_auth_paths, casia_forged_paths, defacto_forged_paths

def load_model(chkpt_path):
    model = MVSSNetLite().to(device)
    chkpt = torch.load(chkpt_path, map_location=device)
    if 'model_state_dict' in chkpt:
        model.load_state_dict(chkpt['model_state_dict'])
    elif 'model' in chkpt:
        model.load_state_dict(chkpt['model'])
    else:
        model.load_state_dict(chkpt)
    model.eval()
    return model

def evaluate_model(model, data_paths, thresholds=[0.5, 0.9]):
    ds = ForgeryDataset([(p['image'], p['mask']) for p in data_paths], is_train=False)
    results = {t: [] for t in thresholds} # list of booleans: True if hallucinated/detected
    
    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            img = ds[i][0].unsqueeze(0).to(device)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            
            for t in thresholds:
                pixels = np.sum(prob > t)
                results[t].append(pixels > 0)
                
    return results

def main():
    print("Loading manifest and building probe sets...")
    casia_auth, casia_forged, defacto_forged = get_probe_sets('reports/manifest.json', 200)
    
    print(f"Extracted {len(casia_auth)} CASIAv2 Authentic")
    print(f"Extracted {len(casia_forged)} CASIAv2 Forged")
    print(f"Extracted {len(defacto_forged)} DEFACTO Forged")
    
    print("Evaluating Epoch 40...")
    model40 = load_model("model/checkpoints/stage1_mvss_lite_ep40.pt")
    res40_auth = evaluate_model(model40, casia_auth)
    res40_c_forged = evaluate_model(model40, casia_forged)
    res40_d_forged = evaluate_model(model40, defacto_forged)
    del model40
    
    print("Evaluating Epoch 45...")
    model45 = load_model("model/checkpoints/stage1_mvss_lite_ep45.pt")
    res45_auth = evaluate_model(model45, casia_auth)
    res45_c_forged = evaluate_model(model45, casia_forged)
    res45_d_forged = evaluate_model(model45, defacto_forged)
    del model45
    
    print("=========================================")
    print("RESULTS COMPARISON")
    print("=========================================")
    
    for t in [0.5, 0.9]:
        print(f"\\n--- THRESHOLD {t} ---")
        
        fp40 = sum(res40_auth[t])
        fp45 = sum(res45_auth[t])
        
        tp40_c = sum(res40_c_forged[t])
        tp45_c = sum(res45_c_forged[t])
        
        tp40_d = sum(res40_d_forged[t])
        tp45_d = sum(res45_d_forged[t])
        
        print(f"Epoch 40 -> CASIA Auth FP: {fp40}/{len(casia_auth)} | CASIA Forged TP: {tp40_c}/{len(casia_forged)} | DEFACTO Forged TP: {tp40_d}/{len(defacto_forged)}")
        print(f"Epoch 45 -> CASIA Auth FP: {fp45}/{len(casia_auth)} | CASIA Forged TP: {tp45_c}/{len(casia_forged)} | DEFACTO Forged TP: {tp45_d}/{len(defacto_forged)}")
        
        if t == 0.9:
            print("\\nFlips for Authentic FP (0.9):")
            for i in range(len(casia_auth)):
                was_fp40 = res40_auth[t][i]
                was_fp45 = res45_auth[t][i]
                if was_fp40 != was_fp45:
                    status = "FP(40)->Correct(45)" if was_fp40 else "Correct(40)->FP(45)"
                    print(f"  [{i:03d}] {os.path.basename(casia_auth[i]['image'])}: {status}")

if __name__ == '__main__':
    main()
