import torch
import cv2
import numpy as np
import os
import json
from tqdm import tqdm

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

device = torch.device('cpu')

def get_probe_sets(manifest_path, max_per_class=200):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    val_set = manifest['val']
    
    casia_auth = []
    casia_forged = []
    defacto_forged = []
    rtm_auth = []
    midv_auth = []
    
    for s in val_set:
        img, mask, dataset = s['image'], s['mask'], s['dataset']
        if dataset == 'CASIAv2':
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if is_forged and len(casia_forged) < max_per_class:
                casia_forged.append(s)
            elif not is_forged and len(casia_auth) < max_per_class:
                casia_auth.append(s)
        elif dataset == 'DEFACTO':
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if is_forged and len(defacto_forged) < max_per_class:
                defacto_forged.append(s)
        elif dataset == 'RTM':
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if not is_forged and len(rtm_auth) < max_per_class:
                rtm_auth.append(s)
        elif dataset == 'MIDV500':
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if not is_forged and len(midv_auth) < max_per_class:
                midv_auth.append(s)
                
    return casia_auth, casia_forged, defacto_forged, rtm_auth, midv_auth

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

def evaluate_model(model, data_paths):
    ds = ForgeryDataset([(p['image'], p['mask']) for p in data_paths], is_train=False)
    results_05 = []
    
    with torch.no_grad():
        for i in tqdm(range(len(ds))):
            img = ds[i][0].unsqueeze(0).to(device)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            pixels_05 = np.sum(prob > 0.5)
            results_05.append(pixels_05 > 0)
                
    return results_05

def main():
    print("Finding checkpoint with lowest val_total_loss...")
    
    chkpt_path = "model/checkpoints/stage2_mvss_lite_ep5.pt"
    if not os.path.exists(chkpt_path):
        print(f"Error: {chkpt_path} does not exist!")
        return
    else:
        print(f"Verified {chkpt_path} exists.")
        
    print("Loading manifest...")
    c_auth, c_forged, d_forged, r_auth, m_auth = get_probe_sets('reports/manifest.json', 200)
    
    print(f"Lengths: CASIA Auth {len(c_auth)}, CASIA Forged {len(c_forged)}, DEFACTO Forged {len(d_forged)}, RTM Auth {len(r_auth)}, MIDV500 Auth {len(m_auth)}")
    
    print("Loading model...")
    model = load_model(chkpt_path)
    
    print("Evaluating CASIA Authentic...")
    res_c_auth = evaluate_model(model, c_auth)
    print("Evaluating CASIA Forged...")
    res_c_forged = evaluate_model(model, c_forged)
    print("Evaluating DEFACTO Forged...")
    res_d_forged = evaluate_model(model, d_forged)
    print("Evaluating RTM Authentic...")
    res_r_auth = evaluate_model(model, r_auth)
    print("Evaluating MIDV500 Authentic...")
    res_m_auth = evaluate_model(model, m_auth)
    
    print("\n\n--- RESULTS at Threshold 0.5 ---")
    print(f"CASIA Authentic FP: {sum(res_c_auth)} / {len(c_auth)}")
    print(f"RTM Authentic FP: {sum(res_r_auth)} / {len(r_auth)}")
    print(f"MIDV500 Authentic FP: {sum(res_m_auth)} / {len(m_auth)}")
    print(f"CASIA Forged TP: {sum(res_c_forged)} / {len(c_forged)}")
    print(f"DEFACTO Forged TP: {sum(res_d_forged)} / {len(d_forged)}")

if __name__ == '__main__':
    main()
