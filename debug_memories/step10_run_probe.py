import torch
import cv2
import numpy as np
import os
import json
import glob
from tqdm import tqdm

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

def get_rtm_probe_sets(manifest_path, max_per_class=200):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    val_set = manifest['val']
    
    rtm_auth = []
    rtm_forged = []
    
    for s in val_set:
        if s['dataset'] == 'RTM':
            img, mask = s['image'], s['mask']
            is_forged = cv2.imread(mask, cv2.IMREAD_GRAYSCALE).sum() > 0
            if is_forged and len(rtm_forged) < max_per_class:
                rtm_forged.append(s)
            elif not is_forged and len(rtm_auth) < max_per_class:
                rtm_auth.append(s)
                
    return rtm_auth, rtm_forged

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

def evaluate_model(model, data_paths, thresholds):
    ds = ForgeryDataset([(p['image'], p['mask']) for p in data_paths], is_train=False)
    
    res_images = {t: 0 for t in thresholds} # count of hallucinated/detected images
    res_pixels = {t: 0 for t in thresholds} # total hallucinated/detected pixels
    
    # Also collect raw predictions for histograms (on the 0.5 threshold pixels)
    # We will just collect a sample of probs for the final checkpoint later.
    
    with torch.no_grad():
        for i in range(len(ds)):
            img = ds[i][0].unsqueeze(0).to(device)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            
            for t in thresholds:
                pixels = np.sum(prob > t)
                if pixels > 0:
                    res_images[t] += 1
                res_pixels[t] += int(pixels)
                
    return res_images, res_pixels

def main():
    rtm_auth, rtm_forged = get_rtm_probe_sets('reports/manifest.json', 200)
    print(f"RTM Auth: {len(rtm_auth)}, RTM Forged: {len(rtm_forged)}")
    
    checkpoints = sorted(glob.glob("model/checkpoints/stage2_mvss_lite_ep*.pt"), 
                         key=lambda x: int(os.path.basename(x).split('ep')[1].split('.pt')[0]))
    
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results = {}
    
    for chkpt in checkpoints:
        ep = int(os.path.basename(chkpt).split('ep')[1].split('.pt')[0])
        print(f"Evaluating Epoch {ep}...")
        model = load_model(chkpt)
        
        auth_imgs, auth_pix = evaluate_model(model, rtm_auth, thresholds)
        forged_imgs, forged_pix = evaluate_model(model, rtm_forged, thresholds)
        
        results[ep] = {
            'auth_fp_imgs': auth_imgs,
            'auth_fp_pix': auth_pix,
            'forged_tp_imgs': forged_imgs,
            'forged_tp_pix': forged_pix
        }
        del model
        
    with open('reports/stage2_probe_results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Done! Saved to reports/stage2_probe_results.json")

if __name__ == '__main__':
    main()
