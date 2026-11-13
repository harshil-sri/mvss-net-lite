"""
Paper 1 — Step 1: RESOLUTION DETECTABILITY PROBE for RTM (read-only, NO training).

Question: RTM (document forgery) is the one domain where NO method helps. The hypothesis
(§0 PENDING, §4b detectability confound) is that 256x256 training destroys tiny document
forgeries before the model can see them. This script quantifies how much forged signal is
recoverable at higher resolution, WITHOUT retraining, so we can decide whether higher-res
training is worth the infra cost BEFORE building it.

Two independent measurements:

  PART A — MASK GEOMETRY (pure, unconfounded, no model).
    For every RTM forged test image, take the NATIVE-resolution mask, find forged connected
    components, and project their bounding boxes through the loader's actual square-resize
    (W->R, H->R) to each target resolution R in {256,512,768,1024}. Report the distribution of
    per-forgery bbox min-side (px) and the fraction of forgeries whose largest component falls
    below detectability floors (<8px ~ below one stride-32 feature cell; <16px). This is an
    UPPER BOUND on what any model could localize at resolution R.

  PART B — MODEL SIGNAL (existing Run4 model, confounded by train/test res mismatch).
    Feed the Run4 checkpoint balanced RTM test images at input resolution R in {256,512,768}
    and measure image-level AUC (segmean, segtopk readouts) + authentic-FP @ matched
    forged-recall. NOTE: Run4 was TRAINED at 256, so higher-res input is out-of-distribution;
    an IMPROVEMENT here is a strong lower bound ("even without retraining, resolution helps"),
    while a flat/negative result is ambiguous (mismatch) and only says "retraining is needed
    to conclude". Interpret directionally.

Read-only w.r.t. model & checkpoints. Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/resolution_probe.py
"""
import os, json, math, random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt'
SEED = int(os.environ.get('RP_SEED', 42))

GEOM_RES = [256, 512, 768, 1024]
MODEL_RES = [int(x) for x in os.environ.get('RP_MODEL_RES', '256,512,768').split(',')]
N_GEOM = int(os.environ.get('RP_NGEOM', 600))          # RTM forged imgs for geometry
N_PER_CLASS = int(os.environ.get('RP_NPERCLASS', 250)) # forged/auth for model probe
TARGET_RECALL = float(os.environ.get('RP_RECALL', 0.85))
VAL_FRAC = 0.4
FLOORS = [8, 16, 32]   # bbox min-side px thresholds


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_rtm_test():
    manifest = json.load(open('reports/manifest.json'))
    flags = json.load(open('reports/forged_flags.json'))['test']
    forged, auth = [], []
    for s in manifest['test']:
        if s['dataset'] != 'RTM':
            continue
        f = flags.get(s['image'])
        if f is None:
            continue
        (forged if f else auth).append((s['image'], s['mask']))
    return forged, auth


# ----------------------------- PART A: geometry -----------------------------
def geometry(forged_pairs):
    rng = random.Random(SEED)
    pairs = forged_pairs[:]
    rng.shuffle(pairs)
    pairs = pairs[:N_GEOM]

    # per resolution: list of (largest-component min-side px), and native fracs
    minside = {r: [] for r in GEOM_RES}
    native_fracs = []
    n_used = 0
    for img_path, mask_path in pairs:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        H, W = m.shape[:2]
        b = (m > 127).astype(np.uint8)
        if b.sum() == 0:
            continue
        native_fracs.append(float(b.sum()) / (H * W))
        n_comp, _, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
        # stats rows: [x,y,w,h,area]; row 0 = background
        comp = stats[1:]
        if comp.shape[0] == 0:
            continue
        # largest forged component by area
        largest = comp[np.argmax(comp[:, 4])]
        cw, ch = int(largest[2]), int(largest[3])
        n_used += 1
        for r in GEOM_RES:
            sx, sy = r / W, r / H          # loader does square resize W->r, H->r
            ms = min(cw * sx, ch * sy)      # projected bbox min-side at res r
            minside[r].append(ms)

    def summ(vals):
        a = np.array(vals, dtype=float)
        return {
            'n': int(a.size),
            'median_minside_px': round(float(np.median(a)), 2),
            'mean_minside_px': round(float(a.mean()), 2),
            'p10_minside_px': round(float(np.percentile(a, 10)), 2),
            **{f'frac_below_{t}px': round(float((a < t).mean()), 4) for t in FLOORS},
        }

    out = {
        'n_forged_used': n_used,
        'native_forged_frac_median': round(float(np.median(native_fracs)), 5),
        'native_forged_frac_mean': round(float(np.mean(native_fracs)), 5),
        'per_resolution': {str(r): summ(minside[r]) for r in GEOM_RES},
    }
    return out


# ----------------------------- PART B: model probe -----------------------------
def build_model():
    model = MVSSNetLite().to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ck.get('model_state_dict', ck.get('model', ck)), strict=False)
    model.eval()
    return model


@torch.no_grad()
def scores_at_res(model, pairs, res):
    """Return {'segmean':[...], 'segtopk':[...]} for the given pairs at input resolution res."""
    ds = ForgeryDataset(pairs, crop_size=res, is_train=False)
    bs = 4 if res <= 512 else 2
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=3)
    segmean, segtopk = [], []
    for imgs, masks, edges in loader:
        imgs = imgs.to(DEVICE)
        seg, edge = model(imgs)
        prob = torch.sigmoid(seg).flatten(1)
        segmean += prob.mean(dim=1).cpu().tolist()
        k = max(1, int(0.01 * prob.shape[1]))
        segtopk += prob.topk(k, dim=1).values.mean(dim=1).cpu().tolist()
    return {'segmean': segmean, 'segtopk': segtopk}


def threshold_at_recall(forged_scores, target_recall):
    s = np.sort(np.array(forged_scores))[::-1]
    k = max(1, int(math.ceil(target_recall * len(s))))
    k = min(k, len(s))
    return float(s[k - 1])


def model_probe(model, forged_pairs, auth_pairs):
    rng = random.Random(SEED)
    f = forged_pairs[:]; a = auth_pairs[:]
    rng.shuffle(f); rng.shuffle(a)
    f = f[:N_PER_CLASS]; a = a[:min(N_PER_CLASS, len(a))]
    # val/test split for FP@matched-recall (threshold fit on val, reported on test)
    nfv, nav = int(len(f) * VAL_FRAC), int(len(a) * VAL_FRAC)
    f_val, f_test = f[:nfv], f[nfv:]
    a_val, a_test = a[:nav], a[nav:]
    print(f"  model probe: forged val/test={len(f_val)}/{len(f_test)} "
          f"auth val/test={len(a_val)}/{len(a_test)}", flush=True)

    out = {}
    for res in MODEL_RES:
        print(f"  [res {res}] scoring...", flush=True)
        fv = scores_at_res(model, f_val, res)
        ft = scores_at_res(model, f_test, res)
        av = scores_at_res(model, a_val, res)
        at = scores_at_res(model, a_test, res)
        entry = {}
        for readout in ('segmean', 'segtopk'):
            ys = [1] * len(ft[readout]) + [0] * len(at[readout])
            auc = float(roc_auc_score(ys, ft[readout] + at[readout])) if ys else None
            thr = threshold_at_recall(fv[readout], TARGET_RECALL)
            rec = float((np.array(ft[readout]) >= thr).mean()) if ft[readout] else None
            fp = float((np.array(at[readout]) >= thr).mean()) if at[readout] else None
            entry[readout] = {'auc': round(auc, 4), 'threshold': round(thr, 4),
                              'recall': round(rec, 4), 'auth_fp': round(fp, 4)}
        out[str(res)] = entry
        for readout in ('segmean', 'segtopk'):
            e = entry[readout]
            print(f"    {readout:>8s}: AUC={e['auc']} | FP@rec{int(TARGET_RECALL*100)}={e['auth_fp']} "
                  f"(rec={e['recall']})", flush=True)
    return out


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE} | seed={SEED} | geom_res={GEOM_RES} | model_res={MODEL_RES}", flush=True)
    forged, auth = load_rtm_test()
    print(f"RTM test: forged={len(forged)} auth={len(auth)}", flush=True)

    print("\n===== PART A: MASK GEOMETRY (unconfounded, no model) =====", flush=True)
    geo = geometry(forged)
    print(f"  native forged-area fraction: median={geo['native_forged_frac_median']} "
          f"mean={geo['native_forged_frac_mean']} (n={geo['n_forged_used']})", flush=True)
    print(f"  {'res':>5s} | {'median':>8s} | {'p10':>7s} | " +
          " | ".join(f'<{t}px' for t in FLOORS), flush=True)
    for r in GEOM_RES:
        s = geo['per_resolution'][str(r)]
        print(f"  {r:>5d} | {s['median_minside_px']:>8.1f} | {s['p10_minside_px']:>7.1f} | " +
              " | ".join(f"{s[f'frac_below_{t}px']:.3f}" for t in FLOORS), flush=True)
    print("  (min-side = projected bbox min side of the LARGEST forged component, in px, "
          "after the loader's square resize. <8px ~ below one stride-32 feature cell.)", flush=True)

    print("\n===== PART B: MODEL SIGNAL (Run4 ckpt, confounded by train@256 mismatch) =====", flush=True)
    model = build_model()
    mp = model_probe(model, forged, auth)

    results = {
        'config': {'seed': SEED, 'geom_res': GEOM_RES, 'model_res': MODEL_RES,
                   'n_geom': N_GEOM, 'n_per_class': N_PER_CLASS,
                   'target_recall': TARGET_RECALL, 'ckpt': CKPT,
                   'note': 'Run4 trained at 256; higher-res model probe is OOD (lower bound only).'},
        'geometry': geo,
        'model_probe': mp,
    }
    os.makedirs('reports', exist_ok=True)
    out = 'reports/resolution_probe.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == '__main__':
    main()
