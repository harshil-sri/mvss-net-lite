import torch
import cv2
import numpy as np
import os
import json
import csv
from tqdm import tqdm

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

device = torch.device('cpu')

def get_probe_sets(manifest_path, max_per_class=200):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    val_set = manifest['val']
    
    casia_auth_paths = []
    
    for s in val_set:
        img, mask, dataset = s['image'], s['mask'], s['dataset']
        if dataset == 'CASIAv2':
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if not is_forged and len(casia_auth_paths) < max_per_class:
                casia_auth_paths.append(s)
                
    return casia_auth_paths

def load_model(chkpt_path):
    model = MVSSNetLite().to(device)
    chkpt = torch.load(chkpt_path, map_location=device)
    if 'model_state_dict' in chkpt:
        model.load_state_dict(chkpt['model_state_dict'])
    elif 'model' in chkpt:
        model.load_state_dict(chkpt['model'])
    else:
        model.load_state_dict(chkpt)
    model.eval() # GUARANTEE EVAL MODE
    return model

def evaluate_model(model, data_paths):
    ds = ForgeryDataset([(p['image'], p['mask']) for p in data_paths], is_train=False) # is_train=False disables augs
    results_09 = []
    results_05 = []
    
    with torch.no_grad(): # GUARANTEE NO GRADIENTS
        for i in range(len(ds)):
            img = ds[i][0].unsqueeze(0).to(device)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            
            pixels_09 = np.sum(prob > 0.9)
            pixels_05 = np.sum(prob > 0.5)
            
            results_09.append(pixels_09 > 0)
            results_05.append(pixels_05 > 0)
                
    return results_09, results_05

def main():
    print("Loading manifest...")
    casia_auth = get_probe_sets('reports/manifest.json', 200)
    
    print("Evaluating Epoch 45 (Run A)...")
    model45 = load_model("model/checkpoints/stage1_mvss_lite_ep45.pt")
    runA_09, runA_05 = evaluate_model(model45, casia_auth)
    
    print("Evaluating Epoch 45 (Run B)...")
    runB_09, runB_05 = evaluate_model(model45, casia_auth)
    
    assert runA_09 == runB_09, "DETERMINISM FAILED FOR 0.9"
    assert runA_05 == runB_05, "DETERMINISM FAILED FOR 0.5"
    print("Determinism successfully verified: Run A and Run B are identical pixel-for-pixel.")
    del model45
    
    print("Evaluating Epoch 40...")
    model40 = load_model("model/checkpoints/stage1_mvss_lite_ep40.pt")
    run40_09, run40_05 = evaluate_model(model40, casia_auth)
    del model40
    
    # Save raw log
    csv_path = "reports/raw_paired_probe_predictions.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'ground_truth', 'ep40_fp_0.9', 'ep45_fp_0.9', 'ep40_fp_0.5', 'ep45_fp_0.5'])
        for i in range(len(casia_auth)):
            writer.writerow([
                casia_auth[i]['image'],
                'authentic',
                run40_09[i],
                runA_09[i],
                run40_05[i],
                runA_05[i]
            ])
            
    print(f"Raw prediction log saved to {csv_path}")

if __name__ == '__main__':
    main()
