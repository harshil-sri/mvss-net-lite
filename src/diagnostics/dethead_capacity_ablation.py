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
DOMAINS = ['RTM', 'CASIAv2']
DOM_ID = {'RTM': 0, 'CASIAv2': 1}
GROUPS = ['RTM_forged', 'RTM_auth', 'CASIAv2_forged', 'CASIAv2_auth']
GID = {g: i for i, g in enumerate(GROUPS)}

STEPS = int(os.environ.get('ABL_STEPS', 2000))
BATCH = 8
LR = 1e-4
POS_WEIGHT = float(os.environ.get('ABL_POSW', 20.0))
DET_LAMBDA = float(os.environ.get('ABL_DETLAMBDA', 1.0))
INIT = os.environ.get('ABL_INIT', 'run4')
SEED = int(os.environ.get('ABL_SEED', 42))
N_EVAL_PER_CELL = int(os.environ.get('ABL_EVAL', 300))
TARGET_RECALL = float(os.environ.get('ABL_RECALL', 0.85))
VAL_FRAC = 0.4
MODE = os.environ.get('ABL_MODE', 'det').lower()


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_pools():
    manifest = json.load(open('reports/manifest.json'))
    flags_all = json.load(open('reports/forged_flags.json'))
    rng = random.Random(SEED)

    train_by_group = defaultdict(list)
    for s in manifest['train']:
        d = s['dataset']
        if d not in DOMAINS:
            continue
        f = flags_all['train'].get(s['image'])
        if f is None:
            continue
        train_by_group[GID[f"{d}_{'forged' if f else 'auth'}"]].append((s['image'], s['mask']))

    cells = defaultdict(list)
    for s in manifest['test']:
        d = s['dataset']
        if d not in DOMAINS:
            continue
        f = flags_all['test'].get(s['image'])
        if f is None:
            continue
        cells[(d, bool(f))].append((s['image'], s['mask']))
    val_cells, test_cells = {}, {}
    for key, lst in cells.items():
        rng.shuffle(lst)
        lst = lst[:N_EVAL_PER_CELL]
        n_val = int(len(lst) * VAL_FRAC)
        val_cells[key] = lst[:n_val]
        test_cells[key] = lst[n_val:]
    return train_by_group, val_cells, test_cells


def build_flat(train_by_group):
    pairs, gids, dids, ylab = [], [], [], []
    for g, lst in train_by_group.items():
        forged = 1 if GROUPS[g].endswith('forged') else 0
        dom = DOM_ID[GROUPS[g].rsplit('_', 1)[0]]
        for (img, mask) in lst:
            pairs.append((img, mask))
            gids.append(g)
            dids.append(dom)
            ylab.append(forged)
    return pairs, np.array(gids), np.array(dids), np.array(ylab, dtype=np.float32)


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

    def forward(self, gap):
        return self.net(gap).squeeze(1)


def per_sample_seg_loss(logits, targets, pos_weight):
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction='none')
    bce = bce.flatten(1).mean(dim=1)
    probs = torch.sigmoid(logits).flatten(1)
    t = targets.flatten(1)
    inter = (probs * t).sum(dim=1)
    region = 1 - (2 * inter + 1e-6) / (probs.sum(dim=1) + t.sum(dim=1) + 1e-6)
    return bce + region


def build_model():
    model = MVSSNetLite().to(DEVICE)
    if INIT == 'run4':
        ck = torch.load(CKPT, map_location=DEVICE)
        model.load_state_dict(ck.get('model_state_dict', ck.get('model', ck)), strict=False)
    return model


def train(pairs, gids, dids, ylab):
    set_seed(SEED)
    rng = random.Random(SEED)
    model = build_model()
    det_mean = DetHead().to(DEVICE)
    det_max = DetHead().to(DEVICE)

    counts = np.bincount(gids, minlength=len(GROUPS)).astype(float)
    counts[counts == 0] = 1
    gw_groups = (1.0 / counts)
    gw_groups = gw_groups / gw_groups.sum() * len(GROUPS)
    gweight = gw_groups[gids]

    if MODE == 'det':
        det_gweight = gweight
        det_ylab = ylab
    elif MODE == 'erm':
        det_gweight = np.ones_like(gids, dtype=float)
        det_ylab = ylab
    elif MODE == 'random':
        rnd_rng = np.random.RandomState(SEED)
        det_gweight = np.ones_like(gids, dtype=float)
        det_ylab = rnd_rng.binomial(1, 0.5, size=len(ylab)).astype(np.float32)
    else:
        raise ValueError(f"Unknown ABL_MODE: {MODE}. Must be one of ['det', 'erm', 'random']")

    f4_store = {}
    hook = model.backbone.fuse4.register_forward_hook(lambda m, i, o: f4_store.__setitem__('f4', o))

    opt = torch.optim.Adam(
        list(model.parameters()) + list(det_mean.parameters()) + list(det_max.parameters()),
        lr=LR
    )

    n = len(pairs)
    batches = [[rng.randrange(n) for _ in range(BATCH)] for _ in range(STEPS)]
    loader = torch.utils.data.DataLoader(IdxDataset(pairs), batch_sampler=batches, num_workers=3)

    model.train()
    det_mean.train()
    det_max.train()
    step = 0
    run_s = 0.0
    run_d = 0.0
    for imgs, masks, idxs in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        idxs = idxs.numpy()
        opt.zero_grad()
        seg, edge = model(imgs)

        seg_loss_per_sample = per_sample_seg_loss(seg, masks, POS_WEIGHT)
        w_seg = torch.tensor([gweight[i] for i in idxs], device=DEVICE)
        seg_loss = (w_seg * seg_loss_per_sample).mean()

        f4 = f4_store['f4']
        gap = f4.flatten(2).mean(dim=2)
        mxp = f4.flatten(2).max(dim=2).values
        y_det = torch.tensor([det_ylab[i] for i in idxs], device=DEVICE)
        w_det = torch.tensor([det_gweight[i] for i in idxs], device=DEVICE)
        det_bce = (
            F.binary_cross_entropy_with_logits(det_mean(gap), y_det, reduction='none')
            + F.binary_cross_entropy_with_logits(det_max(mxp), y_det, reduction='none')
        )
        det_loss = (w_det * det_bce).mean()

        loss = seg_loss + DET_LAMBDA * det_loss
        loss.backward()
        opt.step()
        model.backbone.noise_extractor.normalize_weights()
        run_s += float(seg_loss.detach())
        run_d += float(det_loss.detach())
        step += 1
        if step % 300 == 0:
            print(f"    step {step}/{STEPS} seg={run_s/300:.4f} det={run_d/300:.4f}", flush=True)
            run_s = run_d = 0.0
    hook.remove()
    return model, det_mean, det_max


@torch.no_grad()
def collect_scores(model, det_mean, det_max, cells):
    model.eval()
    det_mean.eval()
    det_max.eval()
    f4_store = {}
    hook = model.backbone.fuse4.register_forward_hook(lambda m, i, o: f4_store.__setitem__('f4', o))
    out = defaultdict(lambda: {'segmean': [], 'dethead_mean': [], 'dethead_max': []})
    for key, pairs in cells.items():
        if not pairs:
            continue
        ds = ForgeryDataset(pairs, is_train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=3)
        for imgs, masks, edges in loader:
            imgs = imgs.to(DEVICE)
            seg, edge = model(imgs)
            prob = torch.sigmoid(seg).flatten(1)
            segmean = prob.mean(dim=1).cpu().tolist()
            f4 = f4_store['f4']
            gap = f4.flatten(2).mean(dim=2)
            mxp = f4.flatten(2).max(dim=2).values
            dm = torch.sigmoid(det_mean(gap)).cpu().tolist()
            dx = torch.sigmoid(det_max(mxp)).cpu().tolist()
            out[key]['segmean'] += segmean
            out[key]['dethead_mean'] += dm
            out[key]['dethead_max'] += dx
    hook.remove()
    return out


def auc_for(readout, scores, domain):
    f = scores[(domain, True)][readout]
    a = scores[(domain, False)][readout]
    if not f or not a:
        return None
    ys = [1] * len(f) + [0] * len(a)
    return float(roc_auc_score(ys, f + a))


def threshold_at_recall(forged_scores, target_recall):
    s = np.sort(np.array(forged_scores))[::-1]
    k = max(1, int(math.ceil(target_recall * len(s))))
    k = min(k, len(s))
    return float(s[k - 1])


def fp_at_matched_recall(readout, val_scores, test_scores, domain, target_recall):
    thr = threshold_at_recall(val_scores[(domain, True)][readout], target_recall)
    f_test = np.array(test_scores[(domain, True)][readout])
    a_test = np.array(test_scores[(domain, False)][readout])
    recall = float((f_test >= thr).mean()) if len(f_test) else None
    fp = float((a_test >= thr).mean()) if len(a_test) else None
    return {
        'threshold': round(thr, 4),
        'recall': round(recall, 4) if recall is not None else None,
        'auth_fp': round(fp, 4) if fp is not None else None
    }


def main():
    if not os.path.exists('reports/forged_flags.json'):
        raise SystemExit("Run diagnostics/cache_forged_flags.py first.")
    print(f"Device: {DEVICE} | mode={MODE} init={INIT} STEPS={STEPS} seed={SEED} pos_weight={POS_WEIGHT} "
          f"det_lambda={DET_LAMBDA} target_recall={TARGET_RECALL}", flush=True)
    train_by_group, val_cells, test_cells = load_pools()
    pairs, gids, dids, ylab = build_flat(train_by_group)

    model, det_mean, det_max = train(pairs, gids, dids, ylab)
    val_scores = collect_scores(model, det_mean, det_max, val_cells)
    test_scores = collect_scores(model, det_mean, det_max, test_cells)

    readouts = ['segmean', 'dethead_mean', 'dethead_max']
    results = {
        'config': {
            'mode': MODE,
            'domains': DOMAINS,
            'init': INIT,
            'steps': STEPS,
            'seed': SEED,
            'pos_weight': POS_WEIGHT,
            'det_lambda': DET_LAMBDA,
            'target_recall': TARGET_RECALL,
            'readouts': readouts
        },
        'per_domain': {}
    }
    print(f"\n===== RESULTS (mode={MODE}, seed={SEED}) =====", flush=True)
    print(f"{'domain':>8s} | {'readout':>12s} |  AUC   | FP@rec{int(TARGET_RECALL*100)} (recall)", flush=True)
    for d in DOMAINS:
        entry = {}
        for readout in readouts:
            auc = auc_for(readout, test_scores, d)
            op = fp_at_matched_recall(readout, val_scores, test_scores, d, TARGET_RECALL)
            entry[readout] = {'auc': round(auc, 4) if auc is not None else None, **op}
            print(f"{d:>8s} | {readout:>12s} | {auc:.4f} | FP={op['auth_fp']} (rec={op['recall']})", flush=True)
        results['per_domain'][d] = entry

    os.makedirs('reports', exist_ok=True)
    out = f"reports/dethead_capacity_{MODE}_seed{SEED}.json"
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == '__main__':
    main()
