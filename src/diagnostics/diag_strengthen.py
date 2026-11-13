"""
Paper 1 — Strengthened diagnosis (Week 1-2 floor).

Extends Action 0 with the evidence a reviewer will demand:

  (A) MULTI-LAYER probe: linear domain-separability at f1,f2,f3,f4 (fused) plus
      n4 (noise branch) and e4 (RGB branch). Shows WHERE domain info concentrates.

  (B) LEARNED-vs-ARCHITECTURAL: run the same probe on THREE model conditions
        - random : all weights reinitialized (no ImageNet, no forgery training)
        - imagenet: ImageNet-pretrained backbone + random heads (the pre-finetune
                    starting point of our model)
        - trained : Run4 ep8_bestval (our forgery-finetuned model)
      If trained >> imagenet >> random at the DECISION layer (f4), the forgery
      training actively organizes features around domain -> shortcut is LEARNED.
      (Honest caveat expected: datasets genuinely differ, so even random/imagenet
       will be > chance. The CLAIM is about the *increase* and about f4 being the
       decision layer feeding both the seg head and the DANN head.)

  (C) STABILITY: 5-fold stratified CV for every probe -> mean +/- std, not a
      single lucky split. Features standardized (StandardScaler) so the three
      model conditions are compared on equal footing.

  (D) MECHANISM: per-domain seg-head authentic-FP vs per-domain TRAIN forged
      base-rate, with Pearson correlation. Tests "the model applies the domain's
      forged prior regardless of content." (n = #domains with authentic test
      samples; reported honestly as directional given small n.)

Read-only w.r.t. model/data. Writes reports/diag_strengthen.json.
Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/diag_strengthen.py
"""
import os, json, random
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

import cv2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt'
DOMAINS = ['RTM', 'MIDV500', 'CASIAv2', 'DEFACTO']
N_PER_DOMAIN = 400          # up from 250
N_BASERATE_SAMPLE = 500     # per-domain train masks sampled for base-rate
BATCH = 8
SEED = 42
CV_FOLDS = 5

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# feature sites to hook: attr on model.backbone -> key
HOOK_SITES = {
    'fuse1': 'f1', 'fuse2': 'f2', 'fuse3': 'f3', 'fuse4': 'f4',
    'noise_layer4': 'n4', 'edge_layer4': 'e4',
}
PROBE_ORDER = ['f1', 'f2', 'f3', 'f4', 'n4', 'e4']


def build_model(condition):
    """condition in {'random','imagenet','trained'}."""
    model = MVSSNetLite().to(DEVICE)  # backbone loads ImageNet weights by default
    if condition == 'random':
        # reinitialize every parameterized module -> no ImageNet, no training
        def reinit(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        model.apply(reinit)
    elif condition == 'imagenet':
        pass  # ImageNet backbone + random heads = pre-finetune starting point
    elif condition == 'trained':
        ckpt = torch.load(CKPT, map_location=DEVICE)
        state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        model.load_state_dict(state, strict=False)
    else:
        raise ValueError(condition)
    model.eval()
    return model


def sample_test_pairs():
    manifest = json.load(open('reports/manifest.json'))
    by_dom = defaultdict(list)
    for s in manifest['test']:
        if s['dataset'] in DOMAINS:
            by_dom[s['dataset']].append((s['image'], s['mask']))
    pairs, dom_labels = [], []
    for d in DOMAINS:
        lst = by_dom[d][:]
        random.shuffle(lst)
        chosen = lst[:N_PER_DOMAIN]
        pairs.extend(chosen)
        dom_labels.extend([d] * len(chosen))
    return pairs, dom_labels


def gap(t):
    return t.flatten(2).mean(dim=2)


def extract(model, loader, dom_labels, collect_seg):
    """Returns feats dict {key: np.array (N,C)} and (if collect_seg) seg firing stats."""
    store = {k: [] for k in HOOK_SITES.values()}
    handles = []
    live = {}
    for attr, key in HOOK_SITES.items():
        mod = getattr(model.backbone, attr)
        handles.append(mod.register_forward_hook(
            lambda m, i, o, k=key: live.__setitem__(k, o.detach())))

    seg_fire_auth = defaultdict(int); seg_fire_forged = defaultdict(int)
    n_auth = defaultdict(int); n_forged = defaultdict(int)

    idx = 0
    with torch.no_grad():
        for imgs, masks, edges in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            seg, edge = model(imgs)
            for key in store:
                store[key].append(gap(live[key]).cpu().numpy())
            if collect_seg:
                prob = torch.sigmoid(seg)
                fired = (prob.flatten(1) > 0.5).any(dim=1).cpu().numpy()
                forged_gt = (masks.flatten(1).sum(dim=1) > 0).cpu().numpy()
                for j in range(imgs.shape[0]):
                    d = dom_labels[idx + j]
                    if forged_gt[j]:
                        n_forged[d] += 1
                        if fired[j]: seg_fire_forged[d] += 1
                    else:
                        n_auth[d] += 1
                        if fired[j]: seg_fire_auth[d] += 1
            idx += imgs.shape[0]
    for h in handles: h.remove()

    feats = {k: np.concatenate(v) for k, v in store.items()}
    seg_stats = None
    if collect_seg:
        seg_stats = {}
        for d in DOMAINS:
            seg_stats[d] = {
                'n_auth': n_auth[d], 'n_forged': n_forged[d],
                'auth_fp_rate': round(seg_fire_auth[d] / max(1, n_auth[d]), 4),
                'forged_recall': round(seg_fire_forged[d] / max(1, n_forged[d]), 4),
            }
    return feats, seg_stats


def cv_probe(X, y):
    """5-fold stratified CV logistic-regression domain accuracy. Returns (mean,std)."""
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=1.0))
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def train_baserates():
    """Per-domain forged base-rate estimated from a sample of the TRAIN split."""
    manifest = json.load(open('reports/manifest.json'))
    by_dom = defaultdict(list)
    for s in manifest['train']:
        if s['dataset'] in DOMAINS:
            by_dom[s['dataset']].append(s['mask'])
    rates = {}
    for d in DOMAINS:
        lst = by_dom[d][:]
        random.shuffle(lst)
        chosen = lst[:N_BASERATE_SAMPLE]
        forged = 0
        for mp in chosen:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None and m.sum() > 0:
                forged += 1
        rates[d] = {'n': len(chosen), 'forged_base_rate': round(forged / max(1, len(chosen)), 4)}
    return rates


def main():
    if not os.path.exists(CKPT):
        raise SystemExit(f"Checkpoint not found: {CKPT}")
    print(f"Device: {DEVICE}")

    pairs, dom_labels = sample_test_pairs()
    print("Sampled per domain:", dict(Counter(dom_labels)))
    ds = ForgeryDataset(pairs, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)
    y = np.array(dom_labels)

    results = {
        'config': {'n_per_domain': N_PER_DOMAIN, 'domains': DOMAINS,
                   'chance': round(1/len(DOMAINS), 4), 'cv_folds': CV_FOLDS,
                   'n_baserate_sample': N_BASERATE_SAMPLE},
        'multi_layer_probe': {},   # condition -> site -> {mean,std}
        'seg_head_firing': {},     # trained only
        'baserate_vs_fp': {},
    }

    for cond in ['random', 'imagenet', 'trained']:
        print(f"\n===== condition: {cond} =====")
        model = build_model(cond)
        feats, seg_stats = extract(model, loader, dom_labels, collect_seg=(cond == 'trained'))
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        cond_out = {}
        for key in PROBE_ORDER:
            mean, std = cv_probe(feats[key], y)
            cond_out[key] = {'acc_mean': round(mean, 4), 'acc_std': round(std, 4)}
            print(f"  [{cond}] {key}: {mean:.3f} +/- {std:.3f}")
        results['multi_layer_probe'][cond] = cond_out

        if seg_stats is not None:
            results['seg_head_firing'] = seg_stats

    # ---- mechanism: base-rate vs auth-FP ----
    print("\n===== base-rate vs authentic-FP =====")
    rates = train_baserates()
    seg = results['seg_head_firing']
    pts = []  # (domain, base_rate, auth_fp)
    for d in DOMAINS:
        br = rates[d]['forged_base_rate']
        auth_fp = seg.get(d, {}).get('auth_fp_rate')
        n_auth = seg.get(d, {}).get('n_auth', 0)
        print(f"  {d:>10s}: train_base_rate={br:.3f} | test_auth_FP="
              f"{auth_fp if auth_fp is not None else 'NA':<6} (n_auth={n_auth})")
        if auth_fp is not None and n_auth > 0:
            pts.append((d, br, auth_fp))

    corr = None
    if len(pts) >= 2:
        brs = np.array([p[1] for p in pts]); fps = np.array([p[2] for p in pts])
        if brs.std() > 0 and fps.std() > 0:
            corr = float(np.corrcoef(brs, fps)[0, 1])
    results['baserate_vs_fp'] = {
        'train_base_rates': rates,
        'points': [{'domain': d, 'base_rate': br, 'auth_fp': fp} for d, br, fp in pts],
        'pearson_r': None if corr is None else round(corr, 4),
        'n_points': len(pts),
        'caveat': 'small n (per-domain); directional evidence, not a clean law',
    }
    print(f"  Pearson r (base-rate vs auth-FP) = "
          f"{'NA' if corr is None else round(corr,3)}  over n={len(pts)} domains")

    os.makedirs('reports', exist_ok=True)
    json.dump(results, open('reports/diag_strengthen.json', 'w'), indent=2)
    print("\nSaved reports/diag_strengthen.json")

    # ---- verdict ----
    print("\n===== VERDICT (f4 decision layer) =====")
    f4 = {c: results['multi_layer_probe'][c]['f4']['acc_mean'] for c in ['random', 'imagenet', 'trained']}
    print(f"  f4 domain acc: random={f4['random']:.3f}  imagenet={f4['imagenet']:.3f}  "
          f"trained={f4['trained']:.3f}  (chance {1/len(DOMAINS):.3f})")
    if f4['trained'] > f4['imagenet'] > f4['random']:
        print("  -> monotone increase: forgery training ACTIVELY amplifies domain "
              "separability at the decision layer (shortcut is LEARNED).")
    elif f4['trained'] > f4['random']:
        print("  -> trained > random but imagenet not strictly between: shortcut "
              "amplified vs random init; report all three honestly.")
    else:
        print("  -> no amplification at f4: reconsider the 'learned' framing.")


if __name__ == '__main__':
    main()
