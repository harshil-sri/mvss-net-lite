import os
import json
import cv2
import torch
import numpy as np
from tqdm import tqdm
import shutil

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(chkpt_path):
    model = MVSSNetLite().to(device)
    chkpt = torch.load(chkpt_path, map_location=device)
    if 'model_state_dict' in chkpt:
        model.load_state_dict(chkpt['model_state_dict'])
    else:
        model.load_state_dict(chkpt)
    model.eval()
    return model

def calculate_iou(pred, gt):
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    if np.sum(union) == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    return np.sum(intersection) / np.sum(union)

def main():
    out_dir = 'reports/verified_docs'
    os.makedirs(out_dir, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model("model/checkpoints/stage2_mvss_lite_ep5.pt")
    
    # Load manifest
    print("Loading manifest...")
    with open('reports/manifest.json', 'r') as f:
        manifest = json.load(f)
        
    val_set = manifest['val']
    
    rtm_auth = []
    rtm_forged = []
    midv_auth = []
    
    # Identify candidates
    for s in val_set:
        dataset = s['dataset']
        mask_path = s['mask']
        is_forged = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).sum() > 0
        if dataset == 'RTM':
            if is_forged:
                rtm_forged.append(s)
            else:
                rtm_auth.append(s)
        elif dataset == 'MIDV500':
            if not is_forged:
                midv_auth.append(s)
                
    # Search for good examples
    def extract_good_examples(candidates, category, target_count, requires_forgery=False):
        ds = ForgeryDataset([(p['image'], p['mask']) for p in candidates], is_train=False)
        found = 0
        
        with torch.no_grad():
            for i in tqdm(range(len(ds)), desc=f"Scanning {category}"):
                if found >= target_count:
                    break
                    
                img_tensor = ds[i][0].unsqueeze(0).to(device)
                gt_mask = ds[i][1].numpy()
                
                _, pred_edge = model(img_tensor)
                prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
                pred_mask = (prob > 0.5).astype(np.uint8)
                
                gt_bin = (gt_mask > 0.5).astype(np.uint8)
                
                is_good = False
                if requires_forgery:
                    iou = calculate_iou(pred_mask, gt_bin)
                    if iou > 0.4:
                        is_good = True
                else:
                    if np.sum(pred_mask) == 0:
                        is_good = True
                        
                if is_good:
                    orig_img_path = candidates[i]['image']
                    orig_mask_path = candidates[i]['mask']
                    
                    base_name = os.path.basename(orig_img_path).split('.')[0]
                    
                    # Read original full size image for saving
                    orig_img = cv2.imread(orig_img_path)
                    gt_img = cv2.imread(orig_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    # Resize pred_mask back to original size
                    h, w = orig_img.shape[:2]
                    pred_mask_resized = cv2.resize(pred_mask * 255, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    # Save
                    cv2.imwrite(os.path.join(out_dir, f"{category}_{base_name}_img.jpg"), orig_img)
                    cv2.imwrite(os.path.join(out_dir, f"{category}_{base_name}_gt.png"), gt_img)
                    cv2.imwrite(os.path.join(out_dir, f"{category}_{base_name}_pred.png"), pred_mask_resized)
                    
                    found += 1
                    
    extract_good_examples(rtm_forged, "RTM_Forged", 5, requires_forgery=True)
    extract_good_examples(rtm_auth, "RTM_Auth", 5, requires_forgery=False)
    extract_good_examples(midv_auth, "MIDV500_Auth", 5, requires_forgery=False)
    
    print(f"\nSaved verified examples to {out_dir}")

if __name__ == '__main__':
    main()
