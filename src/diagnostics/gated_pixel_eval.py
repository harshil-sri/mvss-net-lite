import os, json, random, math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt'
DOMAINS = ['RTM', 'CASIAv2']
DOM_ID = {'RTM': 0, 'CASIAv2': 1}
GROUPS = ['RTM_forged', 'RTM_auth', 'CASIAv2_forged', 'CASIAv2_auth']
GID = {g: i for i, g in enumerate(GROUPS)}

STEPS = int(os.environ.get('GP_STEPS', 2000))
BATCH = 8
LR = 1e-4
POS_WEIGHT = float(os.environ.get('GP_POSW', 20.0))
DET_LAMBDA = float(os.environ.get('GP_DETLAMBDA', 1.0))
INIT = os.environ.get('GP_INIT', 'run4')
SEED = int(os.environ.get('GP_SEED', 42))
N_EVAL_PER_CELL = int(os.environ.get('GP_EVAL', 300))
TARGET_RECALL = float(os.environ.get('GP_RECALL', 0.85))
VAL_FRAC = 0.4


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
        seg_loss = per_sample_seg_loss(seg, masks, POS_WEIGHT).mean()

        f4 = f4_store['f4']
        gap = f4.flatten(2).mean(dim=2)
        mxp = f4.flatten(2).max(dim=2).values
        y = torch.tensor([ylab[i] for i in idxs], device=DEVICE)
        w = torch.tensor([gweight[i] for i in idxs], device=DEVICE)
        det_bce = (
            F.binary_cross_entropy_with_logits(det_mean(gap), y, reduction='none')
            + F.binary_cross_entropy_with_logits(det_max(mxp), y, reduction='none')
        )
        det_loss = (w * det_bce).mean()

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
def collect_image_scores(model, det_max, pairs):
    model.eval()
    det_max.eval()
    f4_store = {}
    hook = model.backbone.fuse4.register_forward_hook(lambda m, i, o: f4_store.__setitem__('f4', o))
    segmeans, detmaxes = [], []
    if pairs:
        ds = ForgeryDataset(pairs, is_train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=3)
        for imgs, masks, edges in loader:
            imgs = imgs.to(DEVICE)
            seg, edge = model(imgs)
            prob = torch.sigmoid(seg).flatten(1)
            segmeans.extend(prob.mean(dim=1).cpu().tolist())
            f4 = f4_store['f4']
            mxp = f4.flatten(2).max(dim=2).values
            detmaxes.extend(torch.sigmoid(det_max(mxp)).cpu().tolist())
    hook.remove()
    return segmeans, detmaxes


def threshold_at_recall(forged_scores, target_recall):
    s = np.sort(np.array(forged_scores))[::-1]
    k = max(1, int(math.ceil(target_recall * len(s))))
    k = min(k, len(s))
    return float(s[k - 1])


@torch.no_grad()
def eval_cell_pixels(model, det_max, pairs, seg_thr, det_thr):
    model.eval()
    det_max.eval()
    f4_store = {}
    hook = model.backbone.fuse4.register_forward_hook(lambda m, i, o: f4_store.__setitem__('f4', o))

    tp_seg, fp_seg, fn_seg = 0, 0, 0
    tp_gate, fp_gate, fn_gate = 0, 0, 0
    total_mask_pos_seg = 0
    total_mask_pos_gate = 0

    if pairs:
        ds = ForgeryDataset(pairs, is_train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=3)
        for imgs, masks, edges in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            seg, edge = model(imgs)
            prob = torch.sigmoid(seg)

            f4 = f4_store['f4']
            mxp = f4.flatten(2).max(dim=2).values
            det_score = torch.sigmoid(det_max(mxp))

            pred_seg = (prob >= seg_thr)
            det_gate = (det_score >= det_thr).view(-1, 1, 1, 1)
            pred_gate = pred_seg & det_gate

            gt = (masks > 0.5)

            tp_seg += int((pred_seg & gt).sum().item())
            fp_seg += int((pred_seg & (~gt)).sum().item())
            fn_seg += int(((~pred_seg) & gt).sum().item())

            tp_gate += int((pred_gate & gt).sum().item())
            fp_gate += int((pred_gate & (~gt)).sum().item())
            fn_gate += int(((~pred_gate) & gt).sum().item())

            total_mask_pos_seg += int(pred_seg.sum().item())
            total_mask_pos_gate += int(pred_gate.sum().item())

    hook.remove()

    def calc_metrics(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        return {
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'iou': round(iou, 4),
            'tp': tp,
            'fp': fp,
            'fn': fn
        }

    return {
        'seg': calc_metrics(tp_seg, fp_seg, fn_seg),
        'gated': calc_metrics(tp_gate, fp_gate, fn_gate),
        'total_mask_pos_seg': total_mask_pos_seg,
        'total_mask_pos_gate': total_mask_pos_gate
    }


def main():
    if not os.path.exists('reports/forged_flags.json'):
        raise SystemExit("Run diagnostics/cache_forged_flags.py first.")
    print(f"Device: {DEVICE} | seed={SEED} pos_weight={POS_WEIGHT} det_lambda={DET_LAMBDA} target_recall={TARGET_RECALL}", flush=True)

    train_by_group, val_cells, test_cells = load_pools()
    pairs, gids, dids, ylab = build_flat(train_by_group)

    model, det_mean, det_max = train(pairs, gids, dids, ylab)

    os.makedirs('reports/logs', exist_ok=True)
    ckpt_out = f"reports/logs/gated_model_seed{SEED}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'det_mean_state_dict': det_mean.state_dict(),
        'det_max_state_dict': det_max.state_dict(),
        'seed': SEED
    }, ckpt_out)
    print(f"Saved weights to {ckpt_out}", flush=True)

    val_forged_casia = val_cells.get(('CASIAv2', True), [])
    val_segmeans, val_detmaxes = collect_image_scores(model, det_max, val_forged_casia)
    seg_thr = threshold_at_recall(val_segmeans, TARGET_RECALL)
    det_thr = threshold_at_recall(val_detmaxes, TARGET_RECALL)

    print(f"Fitted thresholds on CASIAv2 forged val: seg_thr={seg_thr:.4f}, dethead_max_thr={det_thr:.4f}", flush=True)

    casia_test_forged = test_cells.get(('CASIAv2', True), [])
    casia_test_eval = eval_cell_pixels(model, det_max, casia_test_forged, seg_thr, det_thr)

    auth_pixel_counts = {}
    for d in DOMAINS:
        auth_pairs = test_cells.get((d, False), [])
        auth_res = eval_cell_pixels(model, det_max, auth_pairs, seg_thr, det_thr)
        auth_pixel_counts[d] = {
            'fp_pixels_seg': auth_res['total_mask_pos_seg'],
            'fp_pixels_gated': auth_res['total_mask_pos_gate'],
            'n_images': len(auth_pairs)
        }

    results = {
        'config': {
            'seed': SEED,
            'steps': STEPS,
            'target_recall': TARGET_RECALL,
            'seg_threshold': round(seg_thr, 4),
            'dethead_max_threshold': round(det_thr, 4)
        },
        'casia_test_metrics': {
            'seg_only': casia_test_eval['seg'],
            'gated': casia_test_eval['gated']
        },
        'authentic_false_positive_pixels': auth_pixel_counts
    }

    print("\n===== CASIAv2 PIXEL METRICS =====", flush=True)
    print("Seg-only:", results['casia_test_metrics']['seg_only'])
    print("Gated   :", results['casia_test_metrics']['gated'])
    print("\n===== AUTHENTIC FP PIXELS (BEFORE vs AFTER GATING) =====", flush=True)
    for d, counts in auth_pixel_counts.items():
        print(f"{d:>8s}: before={counts['fp_pixels_seg']} after={counts['fp_pixels_gated']} (n={counts['n_images']})")

    out_file = f"reports/gated_pixel_seed{SEED}.json"
    json.dump(results, open(out_file, 'w'), indent=2)
    print(f"\nSaved {out_file}", flush=True)


if __name__ == '__main__':
    main()
