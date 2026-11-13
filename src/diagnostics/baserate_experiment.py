"""
Paper 1 — Controlled base-rate experiment (causal core of the diagnosis).

Claim under test: the model's authentic false-positive rate is CAUSED by the
per-domain forged base-rate in training, not by image content.

Design (one domain = RTM, which has both forged and authentic samples):
  - Reserve a FIXED held-out eval set from the RTM TEST split (never trained on):
      * authentic eval  -> measures authentic-FP
      * forged eval      -> measures forged-recall (sanity/context)
  - Build training sets of a FIXED size N from the RTM TRAIN split at target
    forged base-rates b in {0.0, 0.25, 0.5, 0.75, 1.0}. Everything else is held
    constant across conditions:
      * same common init checkpoint
      * same N, same epochs, same LR, same seed, same eval set
      * FIXED seg pos_weight (does NOT vary with the mix) so the effect cannot be
        attributed to a changing loss weighting -- only the DATA composition varies
  - Fine-tune (seg head only loss) briefly, then measure authentic-FP on the fixed
    authentic eval set.

Prediction: authentic-FP rises monotonically with training base-rate b.
That converts the weak n=3 observational correlation into a controlled, causal result.

Read-only w.r.t. existing checkpoints/manifest (writes only NEW artifacts).
Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/baserate_experiment.py
"""
import os, json, random, argparse, copy
from collections import defaultdict

import numpy as np
import cv2
import torch

from model.network import MVSSNetLite
from model.train import CombinedLoss
from data_pipeline.dataset_loader import ForgeryDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt'
DOMAIN = 'RTM'
SEED = int(os.environ.get('BR_SEED', 42))

# controlled knobs (held constant across conditions)
N_TRAIN = 800
EPOCHS = 4
LR = 1e-4
BATCH = 8
POS_WEIGHT = 100.0          # FIXED across all base-rates (key control)
BASE_RATES = [0.0, 0.25, 0.5, 0.75, 1.0]
N_EVAL_AUTH = 309           # all RTM test authentic
N_EVAL_FORGED = 300


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def is_forged(mask_path):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return m is not None and m.sum() > 0


def split_pools():
    """RTM train pools (forged/auth) + fixed held-out eval sets from RTM test."""
    manifest = json.load(open('reports/manifest.json'))
    train_forged, train_auth = [], []
    for s in manifest['train']:
        if s['dataset'] != DOMAIN:
            continue
        pair = (s['image'], s['mask'])
        (train_forged if is_forged(s['mask']) else train_auth).append(pair)

    test_forged, test_auth = [], []
    for s in manifest['test']:
        if s['dataset'] != DOMAIN:
            continue
        pair = (s['image'], s['mask'])
        (test_forged if is_forged(s['mask']) else test_auth).append(pair)

    rng = random.Random(SEED)
    rng.shuffle(train_forged); rng.shuffle(train_auth)
    rng.shuffle(test_forged); rng.shuffle(test_auth)

    eval_auth = test_auth[:N_EVAL_AUTH]
    eval_forged = test_forged[:N_EVAL_FORGED]
    return train_forged, train_auth, eval_auth, eval_forged


def build_trainset(train_forged, train_auth, b):
    """Fixed-size N_TRAIN set at forged base-rate b (disjoint front-slices)."""
    n_f = round(b * N_TRAIN)
    n_a = N_TRAIN - n_f
    if n_f > len(train_forged) or n_a > len(train_auth):
        raise SystemExit(f"pool too small for b={b}: need {n_f}f/{n_a}a, "
                         f"have {len(train_forged)}f/{len(train_auth)}a")
    pairs = train_forged[:n_f] + train_auth[:n_a]
    random.Random(SEED).shuffle(pairs)
    return pairs, n_f, n_a


def load_init():
    ckpt = torch.load(CKPT, map_location=DEVICE)
    return ckpt.get('model_state_dict', ckpt.get('model', ckpt))


@torch.no_grad()
def eval_fire_rate(model, pairs):
    """Fraction of images where seg fires anywhere (prob>0.5). Image-level."""
    if not pairs:
        return None
    ds = ForgeryDataset(pairs, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)
    model.eval()
    fired_total, n = 0, 0
    for imgs, masks, edges in loader:
        imgs = imgs.to(DEVICE)
        seg, _ = model(imgs)
        prob = torch.sigmoid(seg)
        fired = (prob.flatten(1) > 0.5).any(dim=1)
        fired_total += int(fired.sum().item())
        n += imgs.shape[0]
    return fired_total / max(1, n)


def finetune(init_state, train_pairs):
    set_seed(SEED)  # identical init/order across conditions; only data differs
    model = MVSSNetLite().to(DEVICE)
    model.load_state_dict(init_state, strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    seg_criterion = CombinedLoss(bce_weight=1.0, dice_weight=1.0, pos_weight_val=POS_WEIGHT)

    ds = ForgeryDataset(train_pairs, is_train=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2)
    for ep in range(EPOCHS):
        model.train()
        run = 0.0; nb = 0
        for imgs, masks, edges in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            opt.zero_grad()
            seg, edge = model(imgs)
            loss = seg_criterion(seg, masks)   # seg-only loss (isolate seg head)
            loss.backward()
            opt.step()
            model.backbone.noise_extractor.normalize_weights()
            run += loss.item(); nb += 1
        print(f"    epoch {ep+1}/{EPOCHS} seg_loss={run/max(1,nb):.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--init', choices=['run4', 'imagenet'], default='run4',
                    help="common init: run4 checkpoint (reliable) or imagenet (no forgery prior)")
    args = ap.parse_args()

    if args.init == 'run4' and not os.path.exists(CKPT):
        raise SystemExit(f"Checkpoint not found: {CKPT}")
    print(f"Device: {DEVICE} | init={args.init} | N_train={N_TRAIN} epochs={EPOCHS} "
          f"lr={LR} pos_weight={POS_WEIGHT} (FIXED)")

    train_forged, train_auth, eval_auth, eval_forged = split_pools()
    print(f"RTM train pool: {len(train_forged)} forged / {len(train_auth)} auth")
    print(f"Held-out eval : {len(eval_auth)} auth / {len(eval_forged)} forged (from TEST split)")

    if args.init == 'run4':
        init_state = load_init()
    else:
        init_state = MVSSNetLite().state_dict()  # ImageNet backbone + random heads

    results = {
        'config': {'domain': DOMAIN, 'init': args.init, 'n_train': N_TRAIN,
                   'epochs': EPOCHS, 'lr': LR, 'pos_weight_fixed': POS_WEIGHT,
                   'base_rates': BASE_RATES, 'n_eval_auth': len(eval_auth),
                   'n_eval_forged': len(eval_forged), 'seed': SEED,
                   'metric': 'image-level seg fire@0.5'},
        'conditions': [],
    }

    for b in BASE_RATES:
        print(f"\n===== base-rate b={b} =====")
        train_pairs, n_f, n_a = build_trainset(train_forged, train_auth, b)
        print(f"  train composition: {n_f} forged / {n_a} auth (total {len(train_pairs)})")
        model = finetune(init_state, train_pairs)
        auth_fp = eval_fire_rate(model, eval_auth)
        forged_rec = eval_fire_rate(model, eval_forged)
        print(f"  -> authentic-FP={auth_fp:.4f} | forged-recall={forged_rec:.4f}")
        results['conditions'].append({
            'base_rate': b, 'n_forged': n_f, 'n_auth': n_a,
            'auth_fp': round(auth_fp, 4), 'forged_recall': round(forged_rec, 4),
        })
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    # correlation across the 5 controlled points
    bs = np.array([c['base_rate'] for c in results['conditions']])
    fps = np.array([c['auth_fp'] for c in results['conditions']])
    r = float(np.corrcoef(bs, fps)[0, 1]) if bs.std() > 0 and fps.std() > 0 else None
    # monotonicity (Spearman via rank corr)
    from scipy.stats import spearmanr
    rho, pval = spearmanr(bs, fps)
    results['analysis'] = {
        'pearson_r': None if r is None else round(r, 4),
        'spearman_rho': round(float(rho), 4), 'spearman_p': round(float(pval), 4),
        'monotone_increasing': bool(np.all(np.diff(fps) >= -1e-9)),
    }
    print("\n===== ANALYSIS =====")
    for c in results['conditions']:
        print(f"  b={c['base_rate']:.2f} -> authFP={c['auth_fp']:.3f}  forgedRec={c['forged_recall']:.3f}")
    print(f"  Pearson r={results['analysis']['pearson_r']} | "
          f"Spearman rho={results['analysis']['spearman_rho']} (p={results['analysis']['spearman_p']}) | "
          f"monotone_increasing={results['analysis']['monotone_increasing']}")

    out = f"reports/baserate_experiment_{args.init}_seed{SEED}.json"
    os.makedirs('reports', exist_ok=True)
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved {out}")

    # plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.plot(bs, fps, 'o-', label='authentic-FP', linewidth=2)
        plt.plot(bs, [c['forged_recall'] for c in results['conditions']], 's--',
                 label='forged-recall', alpha=0.7)
        plt.plot([0, 1], [0, 1], ':', color='gray', label='y=x (FP tracks base-rate)')
        plt.xlabel('training forged base-rate'); plt.ylabel('rate on held-out eval')
        plt.title(f'Controlled base-rate experiment (RTM, init={args.init})')
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        p = f"reports/baserate_experiment_{args.init}_seed{SEED}.png"
        plt.savefig(p, dpi=120); plt.close()
        print(f"Saved {p}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == '__main__':
    main()
