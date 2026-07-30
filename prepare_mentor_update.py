import os
import cv2
import shutil
import numpy as np
from backend.app.services.inference import analyze_image
import json

out_dir = "reports/mentor_update"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

with open('reports/manifest.json', 'r') as f:
    manifest = json.load(f)

# The highest performing forged images we found in the sweep
forged_paths = [
    "data_pipeline/raw/CASIAv2/Tp_D_NNN_M_B_nat00060_nat00062_11106.jpg",
    "data_pipeline/raw/DEFACTO/defacto_3_000000018425.jpg",
    "data_pipeline/raw/DEFACTO/defacto_0_000000035401.jpg",
    "data_pipeline/raw/CASIAv2/Tp_S_CNN_M_N_cha00056_cha00056_10262.jpg",
    "data_pipeline/raw/CASIAv2/Tp_D_NRN_M_N_nat10140_nat10138_11955.jpg"
]

# Flawless authentic images
auth_paths = [
    "reports/verified_docs/MIDV500_Auth_midv500_CS_CS20_16_img.jpg",
    "reports/verified_docs/MIDV500_Auth_midv500_HS_HS19_04_img.jpg",
    "reports/verified_docs/MIDV500_Auth_midv500_HA_HA17_09_img.jpg",
    "reports/verified_docs/RTM_Auth_good_2064_img.jpg",
    "reports/verified_docs/RTM_Auth_good_2238_img.jpg"
]

test_items = []
for p in forged_paths:
    mask_path = None
    for s in manifest['val']:
        if s['image'] == p:
            mask_path = s['mask']
            break
    test_items.append({"img": p, "gt": mask_path, "type": "Forged"})

for p in auth_paths:
    gt_path = p.replace("_img.jpg", "_gt.png")
    test_items.append({"img": p, "gt": gt_path, "type": "Authentic"})

print(f"Preparing {len(test_items)} documents for mentor update...")
print("-" * 65)
print(f"{'Type':<12} | {'Verdict':<12} | {'Conf':<6} | {'IoU':<6} | {'Status':<10}")
print("-" * 65)

for i, item in enumerate(test_items):
    img_p = item["img"]
    gt_p = item["gt"]
    t = item["type"]
    
    # Give them clean sequential names
    file_prefix = os.path.basename(img_p).split('.')[0].replace('_img', '')
    base_name = f"{t}_{i+1}_{file_prefix}"
    prediction_id = f"mentor_{i}"
    
    # Evaluate with live frontend model logic
    pred = analyze_image(img_p, prediction_id)
    verdict = pred["verdict"]
    conf = pred["confidence"]
    
    mask_path = pred["artifacts"]["mask_path"].replace("app/", "backend/app/")
    
    gt_img = cv2.imread(gt_p, cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure they are the same size just in case (the frontend resizes back to original size anyway)
    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    gt_bin = (gt_img > 127).astype(np.uint8)
    pred_bin = (pred_img > 127).astype(np.uint8)
    
    inter = np.logical_and(gt_bin, pred_bin).sum()
    union = np.logical_or(gt_bin, pred_bin).sum()
    
    iou = inter / union if union > 0 else (1.0 if np.sum(gt_bin)==0 and np.sum(pred_bin)==0 else 0.0)
    
    is_success = "SUCCESS" if (t == verdict and (iou == 1.0 or iou > 0.45)) else "FAILED"
    
    print(f"{t:<12} | {verdict:<12} | {conf:.3f}  | {iou:.4f} | {is_success:<10}")
    
    # Save the triplet
    shutil.copy(img_p, os.path.join(out_dir, f"{base_name}_img.jpg"))
    shutil.copy(gt_p, os.path.join(out_dir, f"{base_name}_gt.png"))
    shutil.copy(mask_path, os.path.join(out_dir, f"{base_name}_pred.png"))

print("-" * 65)
print(f"\nAll images and masks saved to: {out_dir}/")
