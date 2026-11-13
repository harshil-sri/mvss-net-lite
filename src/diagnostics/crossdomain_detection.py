"""
Paper 1 — Option 1 completion: CROSS-DOMAIN detection head (train sources -> unseen TARGET).

Combines the two option-1 pieces: the detect-then-localize head (4g/4h) evaluated in the
leave-one-domain-out generalization setting (4f) — the practically relevant scenario.

Train on SOURCE = {CASIAv2 (br 0.39, both labels), DEFACTO (br 1.0, forged), MIDV500 (br 0.0, auth)}
with a GROUP-BALANCED image-level detection head (prior-free by construction), evaluate on
held-out TARGET = RTM. Question: does the balanced detection head cut authentic-FP at matched
recall on an UNSEEN domain, vs the seg-mean baseline decision?

Readouts compared on the same model: segmean, segtopk, dethead_mean(GAP), dethead_max(maxpool f4).
Metric on TARGET: image AUC + authentic-FP @ matched forged-recall (threshold fit on a val split
of the target, applied to a disjoint target test split).

Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/crossdomain_detection.py
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

STEPS = int(os.environ.get('CDH_STEPS', 1500))
BATCH = 8
LR = 1e-4
POS_WEIGHT = float(os.environ.get('CDH_POSW', 20.0))
DET_LAMBDA = float(os.environ.get('CDH_DETLAMBDA', 1.0))
INIT = os.environ.get('CDH_INIT', 'run4')
SEED = int(os.environ.get('CDH_SEED', 42))
N_EVAL_PER_CELL = int(os.environ.get('CDH_EVAL', 300))
TARGET_RECALL = float(os.environ.get('CDH_RECALL', 0.85))
VAL_FRAC = 0.4
READOUTS = ['segmean', 'segtopk', 'dethead_mean', 'dethead_max']


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def forged_of(dataset, image, flags):
    if dataset == 'DEFACTO':
        return 1
    if dataset == 'MIDV500':
        return 0
    return flags.get(image)


def load_data():
    manifest = json.load(open('reports/manifest.json'))
    flags_all = json.load(open('reports/forged_flags.json'))
    flags = {**flags_all['train'], **flags_all['test']}
    rng = random.Random(SEED)

    # source train: pairs, group id (domain,label), image forged label
    dom_ids = {d: i for i, d in enumerate(SOURCES)}
    src_pairs, src_group, src_y = [], [], []
    group_names = {}
    for s in manifest['train']:
        d = s['dataset']
        if d not in SOURCES:
            continue
        f = forged_of(d, s['image'], flags)
        if f is None:
            continue
        gname = f"{d}_{'F' if f else 'A'}"
        group_names.setdefault(gname, len(group_names))
        src_pairs.append((s['image'], s['mask']))
        src_group.append(group_names[gname]); src_y.append(float(f))

    # target eval cells (RTM test) -> val + test
    cells = defaultdict(list)
    for s in manifest['test']:
        if s['dataset'] != TARGET:
            continue
        f = forged_of(TARGET, s['image'], flags)
        if f is None:
            continue
        cells[bool(f)].append((s['image'], s['mask']))
    val_cells, test_cells = {}, {}
    for key, lst in cells.items():
        rng.shuffle(lst); lst = lst[:N_EVAL_PER_CELL]
        n_val = int(len(lst) * VAL_FRAC)
        val_cells[key] = lst[:n_val]; test_cells[key] = lst[n_val:]
    return (src_pairs, np.array(src_group), np.array(src_y, dtype=np.float32),
            group_names, val_cells, test_cells)


class IdxDataset(torch.utils.data.Dataset):
    def __init__(self, pairs):
        self.base = ForgeryDataset(pairs, is_train=True)
    def __len__(self):
        return len(self.base)
    def __getitem__(self, i):
        img, mask, edge = self.base[i]
        return img, mask, i


class DetHead(nn.Module):
    def __init__(self, cin=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cin, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
    def forward(self, x):
        return self.net(x).squeeze(1)


def per_sample_seg_loss(logits, targets, pos_weight):
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction='none')
    bce = bce.flatten(1).mean(dim=1)
    probs = torch.sigmoid(logits).flatten(1); t = targets.flatten(1)
    inter = (probs * t).sum(dim=1)
    region = 1 - (2 * inter + 1e-6) / (probs.sum(dim=1) + t.sum(dim=1) + 1e-6)
    return bce + region


def build_model():
    model = MVSSNetLite().to(DEVICE)
    if INIT == 'run4':
        ck = torch.load(CKPT, map_location=DEVICE)
        model.load_state_dict(ck.get('model_state_dict', ck.get('model', ck)), strict=False)
    return model


def train(src_pairs, src_group, src_y, n_groups):
    set_seed(SEED)
    rng = random.Random(SEED)
    model = build_model()
    det_mean = DetHead().to(DEVICE)
    det_max = DetHead().to(DEVICE)

    counts = np.bincount(src_group, minlength=n_groups).astype(float)
    counts[counts == 0] = 1
    gweight = (1.0 / counts); gweight = gweight / gweight.sum() * n_groups

    f4_store = {}
    hook = model.backbone.fuse4.register_forward_hook(lambda m, i, o: f4_store.__setitem__('f4', o))
    opt = torch.optim.Adam(list(model.parameters()) + list(det_mean.parameters())
                           + list(det_max.parameters()), lr=LR)

    n = len(src_pairs)
    batches = [[rng.randrange(n) for _ in range(BATCH)] for _ in range(STEPS)]  # natural sampling
    loader = torch.utils.data.DataLoader(IdxDataset(src_pairs), batch_sampler=batches, num_workers=3)

    model.train(); det_mean.train(); det_max.train(); step = 0; rs = rd = 0.0
    for imgs, masks, idxs in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        idxs = idxs.numpy()
        opt.zero_grad()
        seg, edge = model(imgs)
        seg_loss = per_sample_seg_loss(seg, masks, POS_WEIGHT).mean()
        f4 = f4_store['f4']
        gap = f4.flatten(2).mean(dim=2); mxp = f4.flatten(2).max(dim=2).values
        y = torch.tensor([src_y[i] for i in idxs], device=DEVICE)
        w = torch.tensor([gweight[src_group[i]] for i in idxs], device=DEVICE)
        det_bce = (F.binary_cross_entropy_with_logits(det_mean(gap), y, reduction='none')
                   + F.binary_cross_entropy_with_logits(det_max(mxp), y, reduction='none'))
        det_loss = (w * det_bce).mean()
        loss = seg_loss + DET_LAMBDA * det_loss
        loss.backward(); opt.step()
        model.backbone.noise_extractor.normalize_weights()
        rs += float(seg_loss.detach()); rd += float(det_loss.detach()); step += 1
        if step % 300 == 0:
            print(f"    step {step}/{STEPS} seg={rs/300:.4f} det={rd/300:.4f}", flush=True)
            rs = rd = 0.0
    hook.remove()
    return model, det_mean, det_max


@torch.no_grad()
def collect(model, det_mean, det_max, cells):
    model.eval(); det_mean.eval(); det_max.eval()
    f4_store = {}
    hook = model.backbone.fuse4.register_forward_hook(lambda m, i, o: f4_store.__setitem__('f4', o))
    out = {k: {r: [] for r in READOUTS} for k in cells}
    for key, pairs in cells.items():
        if not pairs:
            continue
        ds = ForgeryDataset(pairs, is_train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=3)
        for imgs, masks, edges in loader:
            imgs = imgs.to(DEVICE)
            seg, edge = model(imgs)
            prob = torch.sigmoid(seg).flatten(1)
            out[key]['segmean'] += prob.mean(dim=1).cpu().tolist()
            k = max(1, int(0.01 * prob.shape[1]))
            out[key]['segtopk'] += prob.topk(k, dim=1).values.mean(dim=1).cpu().tolist()
            f4 = f4_store['f4']
            out[key]['dethead_mean'] += torch.sigmoid(det_mean(f4.flatten(2).mean(dim=2))).cpu().tolist()
            out[key]['dethead_max'] += torch.sigmoid(det_max(f4.flatten(2).max(dim=2).values)).cpu().tolist()
    hook.remove()
    return out


def auc_for(readout, scores):
    f = scores[True][readout]; a = scores[False][readout]
    if not f or not a:
        return None
    return float(roc_auc_score([1] * len(f) + [0] * len(a), f + a))


def threshold_at_recall(forged_scores, target_recall):
    s = np.sort(np.array(forged_scores))[::-1]
    k = min(max(1, int(math.ceil(target_recall * len(s)))), len(s))
    return float(s[k - 1])


def fp_at_matched_recall(readout, val_scores, test_scores, target_recall):
    thr = threshold_at_recall(val_scores[True][readout], target_recall)
    f_test = np.array(test_scores[True][readout]); a_test = np.array(test_scores[False][readout])
    return {'threshold': round(thr, 4),
            'recall': round(float((f_test >= thr).mean()), 4) if len(f_test) else None,
            'auth_fp': round(float((a_test >= thr).mean()), 4) if len(a_test) else None}


def main():
    if not os.path.exists('reports/forged_flags.json'):
        raise SystemExit("Run diagnostics/cache_forged_flags.py first.")
    print(f"Device: {DEVICE} | LOO train={SOURCES} -> test={TARGET} | STEPS={STEPS} init={INIT} "
          f"seed={SEED} pos_weight={POS_WEIGHT} det_lambda={DET_LAMBDA} recall={TARGET_RECALL}", flush=True)
    src_pairs, src_group, src_y, group_names, val_cells, test_cells = load_data()
    print("Source groups:", {g: int((src_group == i).sum()) for g, i in group_names.items()}, flush=True)
    print(f"Target {TARGET}: val {len(val_cells.get(True, []))}F/{len(val_cells.get(False, []))}A | "
          f"test {len(test_cells.get(True, []))}F/{len(test_cells.get(False, []))}A", flush=True)

    model, det_mean, det_max = train(src_pairs, src_group, src_y, len(group_names))
    val_scores = collect(model, det_mean, det_max, val_cells)
    test_scores = collect(model, det_mean, det_max, test_cells)

    results = {'config': {'sources': SOURCES, 'target': TARGET, 'steps': STEPS, 'init': INIT,
                          'seed': SEED, 'pos_weight': POS_WEIGHT, 'det_lambda': DET_LAMBDA,
                          'target_recall': TARGET_RECALL}, 'readouts': {}}
    print(f"\n===== TARGET {TARGET} (unseen) =====", flush=True)
    print(f"{'readout':>12s} |  AUC   | FP@rec{int(TARGET_RECALL*100)} (recall)", flush=True)
    base_fp = base_auc = None
    for r in READOUTS:
        auc = auc_for(r, test_scores)
        op = fp_at_matched_recall(r, val_scores, test_scores, TARGET_RECALL)
        results['readouts'][r] = {'auc': round(auc, 4) if auc is not None else None, **op}
        if r == 'segmean':
            base_fp, base_auc = op['auth_fp'], auc
        print(f"{r:>12s} | {auc:.4f} | FP={op['auth_fp']} (rec={op['recall']})", flush=True)

    print("\n===== vs segmean baseline (lower FP@matched-recall better) =====", flush=True)
    for r in READOUTS:
        if r == 'segmean':
            continue
        e = results['readouts'][r]
        if e['auth_fp'] is not None and base_fp is not None:
            print(f"  {r:>12s}: AUC {e['auc']} ({e['auc']-base_auc:+.3f}) | "
                  f"FP {e['auth_fp']} ({e['auth_fp']-base_fp:+.3f})", flush=True)

    os.makedirs('reports', exist_ok=True)
    out = f"reports/crossdomain_detection_{TARGET}_seed{SEED}.json"
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == '__main__':
    main()
