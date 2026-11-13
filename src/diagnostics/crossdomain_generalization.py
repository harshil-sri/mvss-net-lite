"""
Paper 1 — Cross-domain generalization (leave-one-domain-out). Option-2 key test.

Within-domain eval (4b–4e) does NOT favor domain-robustness methods (they're meant to help
on UNSEEN domains). This is the proper test: train on SOURCE domains with strongly varied
forged base-rates (so the domain->label shortcut is tempting), evaluate on a HELD-OUT TARGET
domain never seen in training.

  SOURCE = CASIAv2 (base-rate ~0.39, both labels), DEFACTO (~1.0, forged-only),
           MIDV500 (~0.0, auth-only)   -> P(forged|domain) varies 0..1 => strong shortcut
  TARGET = RTM (held out; both labels)

Hypothesis: ERM latches onto the source domain->label correlation and generalizes poorly to
RTM; methods that break that correlation (balanced sampling, DANN, decorrelation) should
generalize BETTER to RTM. If so -> that is the paper's positive fix.

Metric on TARGET RTM (test split): per-image AUC (forged vs authentic, threshold-free),
plus FP@0.5, recall@0.5, pixel-IoU.

Assumes DEFACTO=forged / MIDV500=auth by construction (matches measured base-rates 1.0/0.0);
uses reports/forged_flags.json for CASIAv2 + RTM.

Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/crossdomain_generalization.py
"""
import os, json, random, math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt'
SOURCES = ['CASIAv2', 'DEFACTO', 'MIDV500']
TARGET = 'RTM'
SRC_ID = {d: i for i, d in enumerate(SOURCES)}

STEPS = int(os.environ.get('CD_STEPS', 1500))
BATCH = 8
LR = 1e-4
POS_WEIGHT = float(os.environ.get('CD_POSW', 20.0))
DECORR_LAMBDA = float(os.environ.get('CD_DECORR', 10.0))
INIT = os.environ.get('CD_INIT', 'run4')
SEED = int(os.environ.get('CD_SEED', 42))
N_EVAL = int(os.environ.get('CD_EVAL', 300))   # per label cell on TARGET
METHODS = os.environ.get('CD_METHODS', 'erm,balanced,dann,decorr').split(',')


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def forged_of(dataset, image, flags):
    if dataset == 'DEFACTO':
        return 1
    if dataset == 'MIDV500':
        return 0
    return flags.get(image)   # CASIAv2, RTM from cache


def load_data():
    manifest = json.load(open('reports/manifest.json'))
    flags_all = json.load(open('reports/forged_flags.json'))
    flags = {**flags_all['train'], **flags_all['test']}
    rng = random.Random(SEED)

    # SOURCE train pairs + domain id + forged label
    src_pairs, src_dom, src_forged = [], [], []
    for s in manifest['train']:
        d = s['dataset']
        if d not in SOURCES:
            continue
        f = forged_of(d, s['image'], flags)
        if f is None:
            continue
        src_pairs.append((s['image'], s['mask'])); src_dom.append(SRC_ID[d]); src_forged.append(f)

    # TARGET eval cells from RTM test
    tgt_forged, tgt_auth = [], []
    for s in manifest['test']:
        if s['dataset'] != TARGET:
            continue
        f = forged_of(TARGET, s['image'], flags)
        (tgt_forged if f else tgt_auth).append((s['image'], s['mask']))
    rng.shuffle(tgt_forged); rng.shuffle(tgt_auth)
    return (src_pairs, np.array(src_dom), np.array(src_forged),
            tgt_forged[:N_EVAL], tgt_auth[:N_EVAL])


class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.base = ForgeryDataset(pairs, is_train=True)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, mask, edge = self.base[i]
        return img, mask, i


def per_sample_seg_loss(logits, targets, pos_weight):
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction='none')
    bce = bce.flatten(1).mean(dim=1)
    probs = torch.sigmoid(logits).flatten(1); t = targets.flatten(1)
    inter = (probs * t).sum(dim=1)
    region = 1 - (2 * inter + 1e-6) / (probs.sum(dim=1) + t.sum(dim=1) + 1e-6)
    return bce + region


def decorr_multi(f4_gap, dom_ids, K):
    """Penalize covariance between features and each domain one-hot column."""
    onehot = F.one_hot(dom_ids, num_classes=K).float()   # (B,K)
    if onehot.sum(0).min() == 0 and (onehot.sum(0) > 0).sum() < 2:
        return torch.zeros((), device=f4_gap.device)
    f = f4_gap - f4_gap.mean(0, keepdim=True)
    d = onehot - onehot.mean(0, keepdim=True)             # (B,K)
    cov = (f.unsqueeze(2) * d.unsqueeze(1)).mean(0)       # (C,K)
    return (cov ** 2).mean()


def make_group_id(dom, forged):
    return dom * 2 + int(forged)   # unique per (domain,label)


def build_model():
    model = MVSSNetLite().to(DEVICE)
    if INIT == 'run4':
        ck = torch.load(CKPT, map_location=DEVICE)
        model.load_state_dict(ck.get('model_state_dict', ck.get('model', ck)), strict=False)
    # resize domain head to #sources for DANN
    model.domain_classifier[-1] = nn.Linear(256, len(SOURCES)).to(DEVICE)
    return model


def make_batches(src_dom, src_forged, method, rng):
    n = len(src_dom)
    idx_by_group = defaultdict(list)
    for i in range(n):
        idx_by_group[make_group_id(src_dom[i], src_forged[i])].append(i)
    groups = sorted(idx_by_group.keys())
    batches = []
    for _ in range(STEPS):
        if method == 'balanced':
            b = [rng.choice(idx_by_group[rng.choice(groups)]) for _ in range(BATCH)]
        else:
            b = [rng.randrange(n) for _ in range(BATCH)]
        batches.append(b)
    return batches


def train_method(method, src_pairs, src_dom, src_forged):
    set_seed(SEED)
    rng = random.Random(SEED)
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    batches = make_batches(src_dom, src_forged, method, rng)
    loader = torch.utils.data.DataLoader(IdxDataset(src_pairs), batch_sampler=batches, num_workers=3)

    f4_store = {}
    hook = None
    if method == 'decorr':
        hook = model.backbone.fuse4.register_forward_hook(lambda m, i, o: f4_store.__setitem__('f4', o))

    model.train(); step = 0; running = 0.0
    for imgs, masks, idxs in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        idxs = idxs.numpy()
        opt.zero_grad()
        if method == 'dann':
            p = step / float(STEPS)
            lam = (2.0 / (1.0 + math.exp(-10.0 * p))) - 1.0
            seg, edge, dom_logits = model(imgs, lambda_p=lam)
        else:
            seg, edge = model(imgs)
        loss = per_sample_seg_loss(seg, masks, POS_WEIGHT).mean()
        if method == 'dann':
            dom_t = torch.tensor([src_dom[i] for i in idxs], device=DEVICE, dtype=torch.long)
            loss = loss + 0.1 * F.cross_entropy(dom_logits, dom_t)
        if method == 'decorr':
            f4_gap = f4_store['f4'].flatten(2).mean(dim=2)
            dom_t = torch.tensor([src_dom[i] for i in idxs], device=DEVICE, dtype=torch.long)
            loss = loss + DECORR_LAMBDA * decorr_multi(f4_gap, dom_t, len(SOURCES))
        loss.backward(); opt.step()
        model.backbone.noise_extractor.normalize_weights()
        running += float(loss.detach()); step += 1
        if step % 300 == 0:
            print(f"    [{method}] step {step}/{STEPS} loss={running/300:.4f}", flush=True)
            running = 0.0
    if hook:
        hook.remove()
    return model


@torch.no_grad()
def eval_target(model, tgt_forged, tgt_auth):
    model.eval()
    def scores_fire_iou(pairs, is_forged):
        ds = ForgeryDataset(pairs, is_train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=3)
        sc, fired_n, n, iou_sum, iou_n = [], 0, 0, 0.0, 0
        for imgs, masks, edges in loader:
            imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
            seg, edge = model(imgs)
            prob = torch.sigmoid(seg)
            sc += prob.flatten(1).mean(dim=1).cpu().tolist()
            pred = prob > 0.5
            fired_n += int(pred.flatten(1).any(dim=1).sum()); n += imgs.shape[0]
            if is_forged:
                p = pred.flatten(1).float(); t = (masks.flatten(1) > 0.5).float()
                inter = (p * t).sum(1); union = ((p + t) > 0).float().sum(1); v = union > 0
                if v.any():
                    iou_sum += float((inter[v] / union[v]).sum()); iou_n += int(v.sum())
        return sc, fired_n / max(1, n), (iou_sum / max(1, iou_n) if is_forged else None)

    f_sc, f_rec, iou = scores_fire_iou(tgt_forged, True)
    a_sc, a_fp, _ = scores_fire_iou(tgt_auth, False)
    ys = [1] * len(f_sc) + [0] * len(a_sc)
    auc = float(roc_auc_score(ys, f_sc + a_sc))
    return {'target_auc': round(auc, 4), 'target_auth_fp': round(a_fp, 4),
            'target_forged_recall': round(f_rec, 4), 'target_forged_pixel_iou': round(iou, 4)}


def main():
    print(f"Device: {DEVICE} | LOO train={SOURCES} -> test={TARGET} | STEPS={STEPS} "
          f"init={INIT} seed={SEED} pos_weight={POS_WEIGHT}", flush=True)
    src_pairs, src_dom, src_forged, tgt_forged, tgt_auth = load_data()
    for d in SOURCES:
        m = src_dom == SRC_ID[d]
        br = src_forged[m].mean() if m.sum() else float('nan')
        print(f"  source {d}: n={int(m.sum())} forged_base_rate={br:.3f}", flush=True)
    print(f"  target {TARGET} eval: {len(tgt_forged)} forged / {len(tgt_auth)} auth", flush=True)

    results = {'config': {'sources': SOURCES, 'target': TARGET, 'steps': STEPS, 'init': INIT,
                          'seed': SEED, 'pos_weight': POS_WEIGHT, 'decorr_lambda': DECORR_LAMBDA},
               'methods': {}}
    for method in METHODS:
        print(f"\n===== method: {method} =====", flush=True)
        model = train_method(method, src_pairs, src_dom, src_forged)
        ev = eval_target(model, tgt_forged, tgt_auth)
        results['methods'][method] = ev
        print(f"  TARGET {TARGET}: AUC={ev['target_auc']} FP={ev['target_auth_fp']} "
              f"rec={ev['target_forged_recall']} pixIoU={ev['target_forged_pixel_iou']}", flush=True)
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    os.makedirs('reports', exist_ok=True)
    out = f"reports/crossdomain_{TARGET}_seed{SEED}.json"
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved {out}", flush=True)

    print(f"\n===== SUMMARY: generalization to unseen {TARGET} (higher AUC better) =====", flush=True)
    erm_auc = results['methods'].get('erm', {}).get('target_auc')
    for m in METHODS:
        e = results['methods'][m]
        tag = ''
        if erm_auc is not None and m != 'erm':
            d = round(e['target_auc'] - erm_auc, 4)
            tag = f"(Δ vs ERM {d:+}) " + ("IMPROVES" if d > 0.02 else "~same" if abs(d) <= 0.02 else "WORSE")
        print(f"  {m:>10s}: AUC={e['target_auc']} FP={e['target_auth_fp']} "
              f"rec={e['target_forged_recall']} {tag}", flush=True)


if __name__ == '__main__':
    main()
