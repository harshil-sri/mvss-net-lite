import torch
import cv2
import numpy as np
import os
import json
from tqdm import tqdm

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_probe_sets(manifest_path, max_per_class=200):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    val_set = manifest['val']
    
    rtm_auth = []
    midv_auth = []
    
    for s in val_set:
        img, mask, dataset = s['image'], s['mask'], s['dataset']
        if dataset == 'RTM':
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if not is_forged and len(rtm_auth) < max_per_class:
                rtm_auth.append(s)
        elif dataset == 'MIDV500':
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if not is_forged and len(midv_auth) < max_per_class:
                midv_auth.append(s)
                
    return rtm_auth, midv_auth

def load_model(chkpt_path):
    model = MVSSNetLite().to(device)
    chkpt = torch.load(chkpt_path, map_location=device)
    if 'model_state_dict' in chkpt:
        model.load_state_dict(chkpt['model_state_dict'], strict=False)
    elif 'model' in chkpt:
        model.load_state_dict(chkpt['model'], strict=False)
    else:
        model.load_state_dict(chkpt, strict=False)
    model.eval()
    return model

def evaluate_model(model, data_paths, is_dann=True):
    ds = ForgeryDataset([(p['image'], p['mask']) for p in data_paths], is_train=False)
    results_05 = []
    results_09 = []
    
    with torch.no_grad():
        for i in range(len(ds)):
            img = ds[i][0].unsqueeze(0).to(device)
            if is_dann:
                # DANN forward pass returns 3 items
                pred_seg, pred_edge, _ = model(img, lambda_p=0.0)
            else:
                pred_seg, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            
            pixels_05 = np.sum(prob > 0.5)
            pixels_09 = np.sum(prob > 0.9)
            
            results_05.append(pixels_05 > 0)
            results_09.append(pixels_09 > 0)
                
    return sum(results_05), sum(results_09), len(ds)

def main():
    checkpoints = [
        ("model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt", True),
        ("model/checkpoints/stage2run4_mvss_lite_ep25.pt", True),
        ("model/checkpoints/stage2run3_mvss_lite_ep10_bestval.pt", True),
        ("model/checkpoints/stage2run2_mvss_lite_ep40.pt", False),
    ]
    
    r_auth, m_auth = get_probe_sets('reports/manifest.json', 200)
    
    for chkpt, is_dann in checkpoints:
        if not os.path.exists(chkpt):
            print(f"Skipping missing: {chkpt}")
            continue
        print(f"\nEvaluating: {chkpt}")
        model = load_model(chkpt)
        
        rtm_05, rtm_09, rtm_total = evaluate_model(model, r_auth, is_dann)
        midv_05, midv_09, midv_total = evaluate_model(model, m_auth, is_dann)
        
        print(f"RTM-Auth FP (0.5): {rtm_05}/{rtm_total}")
        print(f"RTM-Auth FP (0.9): {rtm_09}/{rtm_total}")
        print(f"MIDV500-Auth FP (0.5): {midv_05}/{midv_total}")
        print(f"MIDV500-Auth FP (0.9): {midv_09}/{midv_total}")

if __name__ == '__main__':
    main()
