import os
import json
import csv
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_checkpoint(path):
    print(f"Loading checkpoint {path}...")
    size = os.path.getsize(path)
    print(f"File size: {size} bytes")
    
    model = MVSSNetLite().to(device)
    chkpt = torch.load(path, map_location=device)
    
    # Report which epoch it corresponds to
    epoch = chkpt.get('epoch', 'Not explicitly saved in dict (inferred from filename)')
    print(f"Epoch encoded in dict: {epoch}")
    
    if 'model_state_dict' in chkpt:
        model.load_state_dict(chkpt['model_state_dict'])
    elif 'model' in chkpt:
        model.load_state_dict(chkpt['model'])
    else:
        model.load_state_dict(chkpt)
    model.eval()
    return model, size, epoch

def generate_overlays(model, items, title, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for i, item in enumerate(items[:10]):
        img_path, mask_path = item['image'], item['mask']
        name = os.path.basename(img_path)
        
        ds = ForgeryDataset([(img_path, mask_path)], is_train=False)
        img_t, mask_t, _ = ds[0]
        
        with torch.no_grad():
            _, pred_edge = model(img_t.unsqueeze(0).to(device))
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            
        orig_img = img_t.permute(1, 2, 0).numpy()
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        orig_img = std * orig_img + mean
        orig_img = np.clip(orig_img, 0, 1)
        
        gt_mask = mask_t.squeeze().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_img)
        axes[0].set_title(f"Original: {name}")
        axes[0].axis('off')
        
        axes[1].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        axes[2].imshow(prob, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title("Predicted Probability")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{title}_{i}.png"))
        plt.close()

def run_probes(model, manifest_path):
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
        
    val_set = manifest['val'] + manifest['test']
    
    probe_items = []
    
    casia_auth = [s for s in val_set if s['dataset'] == 'CASIAv2' and cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE).sum() == 0][:200]
    casia_forged = [s for s in val_set if s['dataset'] == 'CASIAv2' and cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE).sum() > 0][:200]
    defacto_forged = [s for s in val_set if s['dataset'] == 'DEFACTO' and cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE).sum() > 0][:200]
    rtm_auth = [s for s in val_set if s['dataset'] == 'RTM' and cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE).sum() == 0][:200]
    midv_auth = [s for s in val_set if s['dataset'] == 'MIDV500' and cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE).sum() == 0][:200]
    
    # Overlays
    generate_overlays(model, [s for s in val_set if s['dataset'] == 'RTM' and cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE).sum() > 0], "rtm_forged", "reports/raw_overlays")
    generate_overlays(model, rtm_auth, "rtm_auth", "reports/raw_overlays")
    generate_overlays(model, midv_auth, "midv_auth", "reports/raw_overlays")
    generate_overlays(model, casia_auth, "casia_auth", "reports/raw_overlays")
    
    all_probes = [
        ('casia_auth', casia_auth),
        ('casia_forged', casia_forged),
        ('defacto_forged', defacto_forged),
        ('rtm_auth', rtm_auth),
        ('midv_auth', midv_auth)
    ]
    
    results = []
    
    for category, items in all_probes:
        if not items: continue
        ds = ForgeryDataset([(p['image'], p['mask']) for p in items], is_train=False)
        with torch.no_grad():
            for i in range(len(ds)):
                img = ds[i][0].unsqueeze(0).to(device)
                _, pred_edge = model(img)
                prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
                
                is_flagged_05 = int(np.sum(prob > 0.5) > 0)
                is_flagged_09 = int(np.sum(prob > 0.9) > 0)
                
                gt_is_forged = int(cv2.imread(items[i]['mask'], cv2.IMREAD_GRAYSCALE).sum() > 0)
                
                results.append({
                    'image': items[i]['image'],
                    'category': category,
                    'ground_truth_forged': gt_is_forged,
                    'pred_0.5': is_flagged_05,
                    'pred_0.9': is_flagged_09
                })
                
    with open('reports/raw_probe_eval.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'category', 'ground_truth_forged', 'pred_0.5', 'pred_0.9'])
        writer.writeheader()
        writer.writerows(results)

if __name__ == '__main__':
    model, sz, ep = load_checkpoint('model/checkpoints/stage2_mvss_lite_ep50.pt')
    run_probes(model, 'reports/manifest.json')
    print("Done generating raw reports.")
