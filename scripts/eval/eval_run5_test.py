"""Run 5 evaluation: held-out test split, matching Run 4's edge-based detection probe.

For each test image the model's edge prediction is thresholded at 0.5 and 0.9;
an image is flagged forged when any predicted-edge pixel passes the threshold.
Results are reported per dataset and per forgery class so we can see both
forged-recall and authentic-FP behaviour on the final Run 5 model.
"""
import torch
import cv2
import numpy as np
import json
import os
from collections import defaultdict

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def get_test_samples():
    with open('reports/manifest.json', 'r') as f:
        manifest = json.load(f)
    test = manifest['test']
    samples = defaultdict(list)  # datasets in test split
    for s in test:
        if s['dataset'] in ('RTM', 'MIDV500', 'MIDV500_synthetic'):
            samples[s['dataset']].append((s['image'], s['mask']))
    return samples

def evaluate_set(model, pairs, batch=8):
    ds = ForgeryDataset(pairs, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    stats = {
        'n': 0,
        'is_forged': 0,
        'detect_05': 0,
        'detect_09': 0,
        'fp_only_09_forged_0': 0,
    }
    with torch.no_grad():
        for imgs, masks, edges in loader:
            imgs = imgs.to(device, non_blocking=True)
            if isinstance(preds := model(imgs, lambda_p=0.0), tuple):
                _, pred_edge, _ = preds
            else:
                pred_edge, _ = preds
            prob = torch.sigmoid(pred_edge)
            forged_gt = (masks.to(device).sum(dim=(1,2,3)) > 0)
            detect_05 = (prob > 0.5).any(dim=(1,2,3))
            detect_09 = (prob > 0.9).any(dim=(1,2,3))
            stats['n'] += len(imgs)
            stats['is_forged'] += int(forged_gt.sum().item())
            stats['detect_05'] += int((detect_05 & forged_gt).sum().item())
            stats['detect_09'] += int((detect_09 & forged_gt).sum().item())
            stats['fp_only_09_forged_0'] += int(((~forged_gt) & detect_09).sum().item())
    return stats

def main():
    checkpoints = [
        ("model/checkpoints/stage2run5_mvss_lite_ep11_bestval.pt", True, "Run5 ep11_bestval (2.5k iters)"),
        ("model/checkpoints/stage2run5_mvss_lite_ep30.pt", True, "Run5 ep30 (late)"),
        ("model/checkpoints/stage2run5_mvss_lite_ep5.pt", True, "Run5 ep5 (early)"),
    ]
    samples = get_test_samples()
    print("Test-split sizes:", {k: len(v) for k, v in samples.items()})
    out = {}
    for chkpt, is_dann, label in checkpoints:
        if not os.path.exists(chkpt):
            print(f"[SKIP] missing {chkpt}")
            continue
        print(f"\n=== {label} | {chkpt} ===")
        model = load_model(chkpt)
        row = {}
        for ds_name, pairs in samples.items():
            st = evaluate_set(model, pairs)
            forged_recall_05 = st['detect_05'] / max(1, st['is_forged'])
            forged_recall_09 = st['detect_09'] / max(1, st['is_forged'])
            n_auth = st['n'] - st['is_forged']
            fp_09 = st['fp_only_09_forged_0']
            print(f"  {ds_name:>16s}: n={st['n']:5d} forged={st['is_forged']:5d} "
                  f"recall@0.5={forged_recall_05:.3f} recall@0.9={forged_recall_09:.3f} "
                  f"authFP@0.9={fp_09}/{n_auth} ({fp_09/max(1,n_auth)*100:.1f}%)")
            row[ds_name] = {
                'n': st['n'], 'forged': st['is_forged'],
                'recall_05': round(forged_recall_05, 4),
                'recall_09': round(forged_recall_09, 4),
                'authFP_09': fp_09, 'n_auth': n_auth,
            }
        out[label] = row
        del model
        torch.cuda.empty_cache()
    with open('reports/stage2run5_eval.json', 'w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved reports/stage2run5_eval.json")

if __name__ == '__main__':
    main()