"""
Paper 1 — Strengthened diagnosis (Week 1-2 floor) — Path B port.

Extends Action 0 with the evidence a reviewer will demand:

  (A) MULTI-LAYER probe: linear domain-separability at f1,f2,f3,f4 (encoder
      layers). Shows WHERE domain info concentrates.

  (B) LEARNED-vs-ARCHITECTURAL: run the same probe on TWO model conditions
        - random : all weights reinitialized (no ImageNet)
        - imagenet: ImageNet-pretrained backbone + random heads
      If imagenet >> random at the DECISION layer (f4), ImageNet
      pretraining actively organizes features around domain.

  (C) STABILITY: 5-fold stratified CV for every probe -> mean +/- std, not a
      single lucky split. Features standardized (StandardScaler) so the two
      model conditions are compared on equal footing.

Read-only w.r.t. model/data. Writes reports/pathb_diag_strengthen.json.
Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/pathb_diag_strengthen.py
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

from model.pathb_network import ResNetUNetFPN
from data_pipeline.dataset_loader import ForgeryDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DOMAINS = ['RTM', 'MIDV500', 'CASIAv2', 'DEFACTO']
N_PER_DOMAIN = 400
BATCH = 8
SEED = 42
CV_FOLDS = 5

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

HOOK_SITES = {
    'layer1': 'f1', 'layer2': 'f2', 'layer3': 'f3', 'layer4': 'f4',
}
PROBE_ORDER = ['f1', 'f2', 'f3', 'f4']


def build_model(condition):
    model = ResNetUNetFPN().to(DEVICE)
    if condition == 'random':
        def reinit(m):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        model.apply(reinit)
    elif condition == 'imagenet':
        pass  # ImageNet backbone + random heads = controls
    elif condition == 'trained':
        raise ValueError("Path B has no trained forgery checkpoint; use random/imagenet only")
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


def extract(model, loader, dom_labels):
    store = {k: [] for k in HOOK_SITES.values()}
    handles = []
    live = {}
    for attr, key in HOOK_SITES.items():
        mod = getattr(model, attr)
        handles.append(mod.register_forward_hook(
            lambda m, i, o, k=key: live.__setitem__(k, o.detach())))

    with torch.no_grad():
        for imgs, masks, edges in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            model(imgs)
            for key in store:
                store[key].append(gap(live[key]).cpu().numpy())
    for h in handles: h.remove()

    feats = {k: np.concatenate(v) for k, v in store.items()}
    return feats


def cv_probe(X, y):
    """5-fold stratified CV logistic-regression domain accuracy. Returns (mean,std)."""
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=1.0))
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def main():
    print(f"Device: {DEVICE}")

    pairs, dom_labels = sample_test_pairs()
    print("Sampled per domain:", dict(Counter(dom_labels)))
    ds = ForgeryDataset(pairs, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)
    y = np.array(dom_labels)

    results = {
        'config': {'n_per_domain': N_PER_DOMAIN, 'domains': DOMAINS,
                   'chance': round(1/len(DOMAINS), 4), 'cv_folds': CV_FOLDS},
        'multi_layer_probe': {},
    }

    for cond in ['random', 'imagenet']:
        print(f"\n===== condition: {cond} =====")
        model = build_model(cond)
        feats = extract(model, loader, dom_labels)
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        cond_out = {}
        for key in PROBE_ORDER:
            mean, std = cv_probe(feats[key], y)
            cond_out[key] = {'acc_mean': round(mean, 4), 'acc_std': round(std, 4)}
            print(f"  [{cond}] {key}: {mean:.3f} +/- {std:.3f}")
        results['multi_layer_probe'][cond] = cond_out

    os.makedirs('reports', exist_ok=True)
    json.dump(results, open('reports/pathb_diag_strengthen.json', 'w'), indent=2)
    print("\nSaved reports/pathb_diag_strengthen.json")

    print("\n===== VERDICT (f4 decision layer) =====")
    f4 = {c: results['multi_layer_probe'][c]['f4']['acc_mean'] for c in ['random', 'imagenet']}
    print(f"  f4 domain acc: random={f4['random']:.3f}  imagenet={f4['imagenet']:.3f}  "
          f"(chance {1/len(DOMAINS):.3f})")
    if f4['imagenet'] > f4['random']:
        print("  -> imagenet > random: ImageNet pretraining amplifies domain "
              "separability at the decision layer.")
    else:
        print("  -> no amplification at f4: both random and imagenet are similar.")


if __name__ == '__main__':
    main()