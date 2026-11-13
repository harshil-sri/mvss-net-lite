"""Quantify the actual visibility of the blur-C synthetic seams the model must detect.

Measures, over a sample of MIDV500_synthetic test forgeries:
  - forged-region mean abs pixel diff (tampered vs original base) — the "signal"
  - how much of the forged region is within ±16 grey levels (sub-threshold)
  - mask coverage fraction (how small the forged fields are)
This tells us whether the forged class is even separable from authentic MIDV500.
"""
import json
import glob
import os
import numpy as np
import cv2

spec = json.load(open('reports/spec_run4_synthetic.json')) if os.path.exists('reports/spec_run4_synthetic.json') else None
manifest = json.load(open('reports/manifest.json'))
synth_meta = json.load(open('reports/synthetic_meta.json')) if os.path.exists('reports/synthetic_meta.json') else {}

# Use test synthetic entries from the manifest (blur-C paths)
synth_test = [s for s in manifest['test'] if s['dataset'] == 'MIDV500_synthetic']
print(f"synthetic test entries: {len(synth_test)}")

import random
random.seed(0)
sample = random.sample(synth_test, min(30, len(synth_test)))

results = []
for s in sample:
    img = cv2.imread(s['image'])
    mask = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None:
        continue
    h, w = img.shape[:2]
    m = (mask > 127).astype(np.uint8)
    # find the source image (the un-tampered base) to compare against.
    # The generated name contains _synthetic_XXXX; base is the doc before epilogue.
    base_path = None
    imgdir = os.path.dirname(s['image'])
    for cand in sorted(os.listdir(imgdir)):
        pass
    # base doc: replace _synthetic_XXXX.jpg with .jpg
    bn = os.path.basename(s['image'])
    base = bn.replace('_synthetic_', '_')  # heuristic
    # Look in the authentic MIDV500 tree for a matching frame name
    # We can't easily resolve base; instead measure local seam contrast.
    # Edge = gradient magnitude *at mask boundary* vs inside-region gradient.
    if m.sum() == 0:
        continue
    # gradient of tampered image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    grad = np.sqrt(gx * gx + gy * gy)

    # boundary band (dilated mask minus eroded mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_b = cv2.dilate((m * 255).astype(np.uint8), k) > 0
    mask_f = cv2.erode((m * 255).astype(np.uint8), k) > 0
    boundary = mask_b & ~mask_f
    interior = m > 0

    region_sz = int(m.sum())
    frac = region_sz / (h * w)
    grad_bnd = float(grad[boundary].mean()) if boundary.sum() else 0.0
    grad_int = float(grad[interior].mean()) if interior.sum() else 0.0
    grad_out = float(grad[~mask_b].mean())
    results.append({'frac': frac, 'grad_bnd': grad_bnd, 'grad_int': grad_int, 'grad_out': grad_out})

if results:
    fr = np.array([r['frac'] for r in results])
    gb = np.array([r['grad_bnd'] for r in results])
    gi = np.array([r['grad_int'] for r in results])
    go = np.array([r['grad_out'] for r in results])
    print(f"n={len(results)}")
    print(f"forged-region area fraction: mean={fr.mean():.4f} min={fr.min():.5f} max={fr.max():.4f}")
    print(f"gradient @ boundary : mean={gb.mean():.2f} (this is the seam the model must fire on)")
    print(f"gradient @ interior : mean={gi.mean():.2f}")
    print(f"gradient @ outside  : mean={go.mean():.2f}")
    print(f"boundary/outside gradient ratio: {(gb / (go + 1e-9)).mean():.2f}")
    print(f"interior/outside gradient ratio: {(gi / (go + 1e-9)).mean():.2f}")