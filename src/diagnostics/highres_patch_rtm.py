"""
Paper 1 — Step 2: HIGHER-RESOLUTION (512 native patch) TRAINING for RTM.

Motivation (from §4j resolution probe): at 256px ~42% of RTM forgeries are sub-8px (below one
stride-32 feature cell) => physically unresolvable. 512px fixes ~90% of those. The probe also
showed you CANNOT just up-res a 256-trained model at inference (train/test mismatch hurts), so we
must RETRAIN. This script trains under two regimes with an OTHERWISE-IDENTICAL protocol and
compares the RTM image-level ceiling:

  baseline : 256x256 full-image square resize (the current pipeline).
  patch    : 512x512 crops taken at NATIVE resolution (no downscale, no aspect distortion),
             with forgery-aware sampling so a fair fraction of training crops contain edits.

Both: fine-tune from Run4, RTM-only, same STEPS/LR/seed/pos_weight/effective-batch. 512px is fit
into ~8GB via AMP (autocast + GradScaler) + gradient accumulation (micro-batch x ACCUM = eff batch).

EVAL (each model in its own native regime, honest ceiling comparison):
  baseline : single 256 resize -> image score = segmean / segtopk of that image.
  patch    : non-overlapping 512 tiles over the native image -> image score = MAX tile score
             (a forgery lives in one tile; max also gives authentic more chances to fire, which is
             why the primary metric is FP@matched-recall with a calibrated threshold, not fire@0.5).
Metrics per model: image AUC (forged vs authentic) + authentic-FP @ matched forged-recall.

Read-only w.r.t. existing checkpoints (loads Run4, never overwrites it). Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/highres_patch_rtm.py
"""
import os, json, math, random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from model.network import MVSSNetLite

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt'
SEED = int(os.environ.get('HR_SEED', 42))

STEPS = int(os.environ.get('HR_STEPS', 2000))        # optimizer steps per regime
MICRO_BATCH = int(os.environ.get('HR_MICRO', 2))     # samples per forward
ACCUM = int(os.environ.get('HR_ACCUM', 4))           # grad-accum -> eff batch = MICRO*ACCUM
LR = float(os.environ.get('HR_LR', 1e-4))
POS_WEIGHT = float(os.environ.get('HR_POSW', 20.0))
PATCH = int(os.environ.get('HR_PATCH', 512))
BASE_RES = int(os.environ.get('HR_BASERES', 256))
P_FORGED_CENTER = float(os.environ.get('HR_PFC', 0.8))  # prob a forged-img crop is centered on a forgery
N_PER_CLASS = int(os.environ.get('HR_NEVAL', 250))
TARGET_RECALL = float(os.environ.get('HR_RECALL', 0.85))
VAL_FRAC = 0.4
MAX_TILES = int(os.environ.get('HR_MAXTILES', 12))   # cap tiles/image at eval for compute
USE_AMP = os.environ.get('HR_AMP', '1') == '1'
MODES = os.environ.get('HR_MODES', 'baseline,patch').split(',')


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_rtm():
    manifest = json.load(open('reports/manifest.json'))
    flags = json.load(open('reports/forged_flags.json'))
    def collect(split):
        forged, auth = [], []
        fl = flags[split]
        for s in manifest[split]:
            if s['dataset'] != 'RTM':
                continue
            f = fl.get(s['image'])
            if f is None:
                continue
            (forged if f else auth).append((s['image'], s['mask']))
        return forged, auth
    tr_f, tr_a = collect('train')
    te_f, te_a = collect('test')
    return (tr_f, tr_a), (te_f, te_a)


# --------------------------- image / mask IO helpers ---------------------------
def read_img_mask(img_path, mask_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        m = np.zeros(img.shape[:2], np.uint8)
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, (m > 127).astype(np.uint8)


def ensure_min_size(img, mask, size):
    """If either side < size, upscale (keep aspect) so min side == size."""
    H, W = img.shape[:2]
    if H >= size and W >= size:
        return img, mask
    scale = size / min(H, W)
    nW, nH = max(size, int(round(W * scale))), max(size, int(round(H * scale)))
    img = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
    return img, mask


def to_tensors(img, mask):
    edge = (cv2.Canny(mask * 255, 100, 200) > 0).astype(np.uint8)
    it = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    mt = torch.from_numpy(mask).float().unsqueeze(0)
    et = torch.from_numpy(edge).float().unsqueeze(0)
    return it, mt, et


# --------------------------- patch training dataset ---------------------------
class PatchDataset(torch.utils.data.Dataset):
    """Yields PATCH x PATCH native-resolution crops. Forgery-aware for forged images."""
    def __init__(self, pairs, patch, p_forged_center, seed):
        self.pairs = pairs
        self.patch = patch
        self.pfc = p_forged_center
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.pairs)

    def _crop(self, img, mask):
        P = self.patch
        img, mask = ensure_min_size(img, mask, P)
        H, W = img.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(ys) > 0 and self.rng.random() < self.pfc:
            j = self.rng.randrange(len(ys))
            cy, cx = int(ys[j]), int(xs[j])
            top = min(max(cy - P // 2, 0), H - P)
            left = min(max(cx - P // 2, 0), W - P)
        else:
            top = self.rng.randint(0, H - P)
            left = self.rng.randint(0, W - P)
        return img[top:top + P, left:left + P], mask[top:top + P, left:left + P]

    def __getitem__(self, i):
        img, mask = read_img_mask(*self.pairs[i])
        ci, cm = self._crop(img, mask)
        if self.rng.random() < 0.5:            # light aug: horizontal flip
            ci, cm = ci[:, ::-1].copy(), cm[:, ::-1].copy()
        return to_tensors(ci, cm)


class ResizeDataset(torch.utils.data.Dataset):
    """Baseline: full-image square resize to BASE_RES (matches current pipeline)."""
    def __init__(self, pairs, res, seed):
        self.pairs = pairs
        self.res = res
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img, mask = read_img_mask(*self.pairs[i])
        R = self.res
        img_r = cv2.resize(img, (R, R), interpolation=cv2.INTER_LINEAR)
        mt = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        mask_r = F.adaptive_max_pool2d(mt, (R, R)).squeeze().numpy().astype(np.uint8)
        if self.rng.random() < 0.5:
            img_r, mask_r = img_r[:, ::-1].copy(), mask_r[:, ::-1].copy()
        return to_tensors(img_r, mask_r)


def seg_loss_fn(logits, targets, pos_weight):
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
    probs = torch.sigmoid(logits).flatten(1); t = targets.flatten(1)
    inter = (probs * t).sum(dim=1)
    dice = 1 - (2 * inter + 1e-6) / (probs.sum(dim=1) + t.sum(dim=1) + 1e-6)
    return bce + dice.mean()


def build_model():
    model = MVSSNetLite().to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ck.get('model_state_dict', ck.get('model', ck)), strict=False)
    return model


def train_regime(mode, train_pairs):
    set_seed(SEED)
    model = build_model()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    if mode == 'patch':
        ds = PatchDataset(train_pairs, PATCH, P_FORGED_CENTER, SEED)
    else:
        ds = ResizeDataset(train_pairs, BASE_RES, SEED)

    n = len(ds)
    rng = random.Random(SEED)
    total_micro = STEPS * ACCUM
    order = [[rng.randrange(n) for _ in range(MICRO_BATCH)] for _ in range(total_micro)]
    loader = torch.utils.data.DataLoader(ds, batch_sampler=order, num_workers=3)

    print(f"  [{mode}] training: steps={STEPS} eff_batch={MICRO_BATCH*ACCUM} "
          f"(micro={MICRO_BATCH} x accum={ACCUM}) amp={USE_AMP} n_pairs={n}", flush=True)
    opt.zero_grad(set_to_none=True)
    run = 0.0; step = 0; micro = 0
    for imgs, masks, edges in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            seg, edge = model(imgs)
            loss = seg_loss_fn(seg, masks, POS_WEIGHT) / ACCUM
        scaler.scale(loss).backward()
        run += float(loss.detach()) * ACCUM
        micro += 1
        if micro % ACCUM == 0:
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            model.backbone.noise_extractor.normalize_weights()
            step += 1
            if step % 200 == 0:
                print(f"    step {step}/{STEPS} seg={run/(200*ACCUM):.4f}", flush=True)
                run = 0.0
    return model


# --------------------------- eval ---------------------------
@torch.no_grad()
def score_image_resize(model, img, mask, res):
    R = res
    img_r = cv2.resize(img, (R, R), interpolation=cv2.INTER_LINEAR)
    it = torch.from_numpy(img_r.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.amp.autocast('cuda', enabled=USE_AMP):
        seg, _ = model(it)
    prob = torch.sigmoid(seg).flatten()
    k = max(1, int(0.01 * prob.numel()))
    return float(prob.mean()), float(prob.topk(k).values.mean())


@torch.no_grad()
def score_image_tiled(model, img, mask, patch, max_tiles):
    img, mask = ensure_min_size(img, mask, patch)
    H, W = img.shape[:2]
    ys = list(range(0, max(1, H - patch + 1), patch)) or [0]
    xs = list(range(0, max(1, W - patch + 1), patch)) or [0]
    if ys[-1] != H - patch:
        ys.append(H - patch)
    if xs[-1] != W - patch:
        xs.append(W - patch)
    coords = [(y, x) for y in ys for x in xs][:max_tiles]
    tiles = [img[y:y + patch, x:x + patch] for (y, x) in coords]
    batch = torch.from_numpy(np.stack(tiles).transpose(0, 3, 1, 2)).float().to(DEVICE) / 255.0
    means, topks = [], []
    for s in range(0, len(batch), 4):
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            seg, _ = model(batch[s:s + 4])
        prob = torch.sigmoid(seg).flatten(1)
        k = max(1, int(0.01 * prob.shape[1]))
        means += prob.mean(dim=1).cpu().tolist()
        topks += prob.topk(k, dim=1).values.mean(dim=1).cpu().tolist()
    return max(means), max(topks)   # image score = max over tiles


def collect_scores(model, pairs, mode):
    seg_m, seg_t = [], []
    for img_path, mask_path in pairs:
        img, mask = read_img_mask(img_path, mask_path)
        if mode == 'patch':
            m, t = score_image_tiled(model, img, mask, PATCH, MAX_TILES)
        else:
            m, t = score_image_resize(model, img, mask, BASE_RES)
        seg_m.append(m); seg_t.append(t)
    return {'segmean': seg_m, 'segtopk': seg_t}


def threshold_at_recall(forged_scores, target_recall):
    s = np.sort(np.array(forged_scores))[::-1]
    k = min(max(1, int(math.ceil(target_recall * len(s)))), len(s))
    return float(s[k - 1])


def evaluate(model, mode, test_forged, test_auth):
    rng = random.Random(SEED)
    f = test_forged[:]; a = test_auth[:]
    rng.shuffle(f); rng.shuffle(a)
    f = f[:N_PER_CLASS]; a = a[:min(N_PER_CLASS, len(a))]
    nfv, nav = int(len(f) * VAL_FRAC), int(len(a) * VAL_FRAC)
    f_val, f_test = f[:nfv], f[nfv:]
    a_val, a_test = a[:nav], a[nav:]

    fv = collect_scores(model, f_val, mode); ft = collect_scores(model, f_test, mode)
    av = collect_scores(model, a_val, mode); at = collect_scores(model, a_test, mode)
    entry = {}
    for readout in ('segmean', 'segtopk'):
        ys = [1] * len(ft[readout]) + [0] * len(at[readout])
        auc = float(roc_auc_score(ys, ft[readout] + at[readout]))
        thr = threshold_at_recall(fv[readout], TARGET_RECALL)
        rec = float((np.array(ft[readout]) >= thr).mean())
        fp = float((np.array(at[readout]) >= thr).mean())
        entry[readout] = {'auc': round(auc, 4), 'threshold': round(thr, 4),
                          'recall': round(rec, 4), 'auth_fp': round(fp, 4)}
    return entry


def main():
    if not os.path.exists('reports/forged_flags.json'):
        raise SystemExit("Run diagnostics/cache_forged_flags.py first.")
    print(f"Device: {DEVICE} | seed={SEED} steps={STEPS} patch={PATCH} base={BASE_RES} "
          f"amp={USE_AMP} modes={MODES}", flush=True)
    (tr_f, tr_a), (te_f, te_a) = load_rtm()
    train_pairs = tr_f + tr_a
    print(f"RTM train: forged={len(tr_f)} auth={len(tr_a)} | test: forged={len(te_f)} auth={len(te_a)}",
          flush=True)

    results = {'config': {'seed': SEED, 'steps': STEPS, 'patch': PATCH, 'base_res': BASE_RES,
                          'micro_batch': MICRO_BATCH, 'accum': ACCUM, 'lr': LR,
                          'pos_weight': POS_WEIGHT, 'p_forged_center': P_FORGED_CENTER,
                          'target_recall': TARGET_RECALL, 'amp': USE_AMP, 'ckpt': CKPT},
               'regimes': {}}
    for mode in MODES:
        print(f"\n===== REGIME: {mode} =====", flush=True)
        model = train_regime(mode, train_pairs)
        entry = evaluate(model, mode, te_f, te_a)
        results['regimes'][mode] = entry
        for readout in ('segmean', 'segtopk'):
            e = entry[readout]
            print(f"  {mode}/{readout}: AUC={e['auc']} | FP@rec{int(TARGET_RECALL*100)}={e['auth_fp']} "
                  f"(rec={e['recall']})", flush=True)
        del model
        torch.cuda.empty_cache()

    print("\n===== SUMMARY: patch(512) vs baseline(256) on RTM =====", flush=True)
    if 'baseline' in results['regimes'] and 'patch' in results['regimes']:
        for readout in ('segmean', 'segtopk'):
            b = results['regimes']['baseline'][readout]
            p = results['regimes']['patch'][readout]
            print(f"  {readout}: AUC {b['auc']} -> {p['auc']} ({p['auc']-b['auc']:+.3f}) | "
                  f"FP {b['auth_fp']} -> {p['auth_fp']} ({p['auth_fp']-b['auth_fp']:+.3f})", flush=True)

    os.makedirs('reports', exist_ok=True)
    out = f"reports/highres_patch_rtm_seed{SEED}.json"
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == '__main__':
    main()
