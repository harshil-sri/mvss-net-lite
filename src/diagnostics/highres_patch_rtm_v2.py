"""
Paper 1 — Step 2 / ATTEMPT 2: controlled RESOLUTION ablation for RTM, patch-level readout.

Why v2 (see §4k): attempt 1 failed for diagnosed reasons — (a) image-level MAX-over-tiles
aggregation saturated because the model fires on any text, and (b) forgery-centered training had
NO hard negatives, so the model learned "text -> forged" (carpet-bombing). Neither is evidence
against the §4j geometry (forged signal physically exists at 512px); both are readout/training bugs.

v2 fixes both and isolates the pure resolution variable with a CONTROLLED ablation:
  - Sampling is IDENTICAL across regimes (same crops, same seed). Two regimes:
        r512 : feed the 512x512 native crop.
        r256 : feed the SAME crop downsampled to 256x256 (only pixel density differs).
    This removes the full-document-downscale and aspect-distortion confounds — the only difference
    between regimes is resolution of the same content.
  - BALANCED HARD NEGATIVES in training: patches are 50% forged (crop centered on a forgery),
    25% hard-negative authentic text (empty-mask crop from a FORGED doc), 25% authentic (crop from
    an authentic doc). The model must learn forged-text vs authentic-text, not text vs blank.
  - PATCH/PIXEL-level metrics (no image-max):
        PRIMARY  pixel-level localization on forged patches: pixel-AUC + best pixel-F1 (forged vs
                 non-forged pixels). Immune to aggregation confounds; the honest localization metric.
        SECONDARY patch-level detection: patch score = top-1% pixel prob (segtopk); forged-patch vs
                 authentic-patch AUC + authentic-FP @ matched recall.

Fine-tunes from Run4, RTM-only, seg loss with hard negatives. 512 fits ~8GB via AMP + grad-accum.
Read-only w.r.t. existing checkpoints. Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/highres_patch_rtm_v2.py
"""
import os, json, math, random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_curve

from model.network import MVSSNetLite

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt'
SEED = int(os.environ.get('HR_SEED', 42))

STEPS = int(os.environ.get('HR_STEPS', 2000))
MICRO_BATCH = int(os.environ.get('HR_MICRO', 2))
ACCUM = int(os.environ.get('HR_ACCUM', 4))
LR = float(os.environ.get('HR_LR', 1e-4))
POS_WEIGHT = float(os.environ.get('HR_POSW', 20.0))
PATCH = int(os.environ.get('HR_PATCH', 512))
P_FORGED = float(os.environ.get('HR_PFORGED', 0.5))     # frac of train patches that are forged
P_HARDNEG = float(os.environ.get('HR_PHARDNEG', 0.25))  # frac that are hard-neg (auth text in forged doc)
N_EVAL_FORGED = int(os.environ.get('HR_NEVALF', 200))
N_EVAL_AUTH = int(os.environ.get('HR_NEVALA', 200))
TARGET_RECALL = float(os.environ.get('HR_RECALL', 0.85))
USE_AMP = os.environ.get('HR_AMP', '1') == '1'
REGIMES = os.environ.get('HR_REGIMES', 'r256,r512').split(',')
MAX_NEG_PER_PATCH = int(os.environ.get('HR_MAXNEG', 3000))  # pixel-metric negative cap per forged patch


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_rtm():
    manifest = json.load(open('reports/manifest.json'))
    flags = json.load(open('reports/forged_flags.json'))
    def collect(split):
        fo, au = [], []
        fl = flags[split]
        for s in manifest[split]:
            if s['dataset'] != 'RTM':
                continue
            f = fl.get(s['image'])
            if f is None:
                continue
            (fo if f else au).append((s['image'], s['mask']))
        return fo, au
    return collect('train'), collect('test')


def read_img_mask(img_path, mask_path):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        m = np.zeros(img.shape[:2], np.uint8)
    if m.shape[:2] != img.shape[:2]:
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, (m > 127).astype(np.uint8)


def ensure_min_size(img, mask, size):
    H, W = img.shape[:2]
    if H >= size and W >= size:
        return img, mask
    scale = size / min(H, W)
    nW, nH = max(size, int(round(W * scale))), max(size, int(round(H * scale)))
    img = cv2.resize(img, (nW, nH), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (nW, nH), interpolation=cv2.INTER_NEAREST)
    return img, mask


def forged_crop(img, mask, P, rng):
    """512 crop centered on a random forged pixel (guaranteed to contain forgery)."""
    img, mask = ensure_min_size(img, mask, P)
    H, W = img.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    j = rng.randrange(len(ys)); cy, cx = int(ys[j]), int(xs[j])
    top = min(max(cy - P // 2, 0), H - P); left = min(max(cx - P // 2, 0), W - P)
    return img[top:top + P, left:left + P], mask[top:top + P, left:left + P]


def empty_crop(img, mask, P, rng, tries=8):
    """512 crop whose mask is empty (authentic-text hard negative). Falls back to any crop."""
    img, mask = ensure_min_size(img, mask, P)
    H, W = img.shape[:2]
    best = None
    for _ in range(tries):
        top = rng.randint(0, H - P); left = rng.randint(0, W - P)
        cm = mask[top:top + P, left:left + P]
        best = (img[top:top + P, left:left + P], cm)
        if cm.sum() == 0:
            return best
    return best   # accept least-forged attempt if none fully empty


def maybe_downsample(img, mask, regime):
    if regime == 'r512':
        return img, mask
    # r256: resize the SAME crop to 256 (isolates resolution). mask via max-pool to keep positives.
    img256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    mt = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    m256 = F.adaptive_max_pool2d(mt, (256, 256)).squeeze().numpy().astype(np.uint8)
    return img256, m256


def to_tensors(img, mask):
    it = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    mt = torch.from_numpy(mask).float().unsqueeze(0)
    return it, mt


class TrainPatches(torch.utils.data.Dataset):
    """Balanced patch sampler with hard negatives. Deterministic per index via seeded rng."""
    def __init__(self, forged_pairs, auth_pairs, regime, seed):
        self.fp = forged_pairs; self.ap = auth_pairs
        self.regime = regime; self.seed = seed

    def __len__(self):
        return 10_000_000   # virtual; bounded by batch_sampler length

    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)   # identical crops across regimes for same i
        r = rng.random()
        for _ in range(6):
            if r < P_FORGED:                       # forged patch
                img, mask = read_img_mask(*self.fp[rng.randrange(len(self.fp))])
                crop = forged_crop(img, mask, PATCH, rng)
            elif r < P_FORGED + P_HARDNEG:         # hard-neg: authentic text in a forged doc
                img, mask = read_img_mask(*self.fp[rng.randrange(len(self.fp))])
                crop = empty_crop(img, mask, PATCH, rng)
            else:                                   # authentic doc
                img, mask = read_img_mask(*self.ap[rng.randrange(len(self.ap))])
                crop = empty_crop(img, mask, PATCH, rng)
            if crop is not None:
                break
            r = rng.random()
        ci, cm = crop
        if rng.random() < 0.5:
            ci, cm = ci[:, ::-1].copy(), cm[:, ::-1].copy()
        ci, cm = maybe_downsample(ci, cm, self.regime)
        return to_tensors(ci, cm)


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


def train_regime(regime, forged_pairs, auth_pairs):
    set_seed(SEED)
    model = build_model(); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    ds = TrainPatches(forged_pairs, auth_pairs, regime, SEED)
    total_micro = STEPS * ACCUM
    loader = torch.utils.data.DataLoader(ds, batch_size=MICRO_BATCH, num_workers=3,
                                         sampler=range(total_micro * MICRO_BATCH))
    print(f"  [{regime}] training: steps={STEPS} eff_batch={MICRO_BATCH*ACCUM} amp={USE_AMP} "
          f"forged/hardneg/auth={P_FORGED}/{P_HARDNEG}/{round(1-P_FORGED-P_HARDNEG,2)}", flush=True)
    opt.zero_grad(set_to_none=True); run = 0.0; step = 0; micro = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            seg, edge = model(imgs)
            loss = seg_loss_fn(seg, masks, POS_WEIGHT) / ACCUM
        scaler.scale(loss).backward()
        run += float(loss.detach()) * ACCUM; micro += 1
        if micro % ACCUM == 0:
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            model.backbone.noise_extractor.normalize_weights(); step += 1
            if step % 200 == 0:
                print(f"    step {step}/{STEPS} seg={run/(200*ACCUM):.4f}", flush=True); run = 0.0
            if step >= STEPS:
                break
    return model


# ------------------------------- eval -------------------------------
def build_eval_patches(forged_test, auth_test, rng):
    """Fixed patch set (native 512 crops + masks), reused across regimes."""
    fpatch, apatch = [], []
    ftp = forged_test[:]; atp = auth_test[:]
    rng.shuffle(ftp); rng.shuffle(atp)
    # forged patches
    for pair in ftp:
        if len(fpatch) >= N_EVAL_FORGED:
            break
        img, mask = read_img_mask(*pair)
        c = forged_crop(img, mask, PATCH, rng)
        if c is not None and c[1].sum() > 0:
            fpatch.append(c)
    # authentic patches: half hard-neg (forged docs, empty), half authentic docs
    n_hn = N_EVAL_AUTH // 2
    for pair in ftp:
        if len([1 for _ in apatch]) >= n_hn:
            break
        img, mask = read_img_mask(*pair)
        c = empty_crop(img, mask, PATCH, rng)
        if c is not None and c[1].sum() == 0:
            apatch.append(c)
    for pair in atp:
        if len(apatch) >= N_EVAL_AUTH:
            break
        img, mask = read_img_mask(*pair)
        c = empty_crop(img, mask, PATCH, rng)
        if c is not None and c[1].sum() == 0:
            apatch.append(c)
    return fpatch, apatch


@torch.no_grad()
def eval_regime(model, regime, fpatch, apatch, rng):
    model.eval()
    # pixel-level accumulation (forged patches only) + patch-level scores
    px_prob, px_lab = [], []
    f_scores, a_scores = [], []

    def run_patch(crop):
        img, mask = maybe_downsample(crop[0], crop[1], regime)
        it = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            seg, _ = model(it)
        prob = torch.sigmoid(seg).squeeze().float().cpu().numpy()   # HxW
        return prob, mask

    for crop in fpatch:
        prob, mask = run_patch(crop)
        p = prob.ravel(); m = mask.ravel().astype(np.uint8)
        pos_idx = np.where(m == 1)[0]; neg_idx = np.where(m == 0)[0]
        if len(pos_idx) == 0:
            continue
        if len(neg_idx) > MAX_NEG_PER_PATCH:
            neg_idx = rng.sample(list(neg_idx), MAX_NEG_PER_PATCH)
        idx = np.concatenate([pos_idx, np.array(neg_idx, dtype=int)])
        px_prob.append(p[idx]); px_lab.append(m[idx])
        k = max(1, int(0.01 * p.size))
        f_scores.append(float(np.sort(p)[::-1][:k].mean()))   # top-1% = patch forged score
    for crop in apatch:
        prob, mask = run_patch(crop)
        p = prob.ravel(); k = max(1, int(0.01 * p.size))
        a_scores.append(float(np.sort(p)[::-1][:k].mean()))

    # pixel-level metrics
    P = np.concatenate(px_prob); L = np.concatenate(px_lab)
    pix_auc = float(roc_auc_score(L, P)) if L.min() != L.max() else None
    prec, rec, thr = precision_recall_curve(L, P)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    pix_bestf1 = float(np.nanmax(f1))

    # patch-level detection
    ys = [1] * len(f_scores) + [0] * len(a_scores)
    patch_auc = float(roc_auc_score(ys, f_scores + a_scores))
    s = np.sort(np.array(f_scores))[::-1]
    k = min(max(1, int(math.ceil(TARGET_RECALL * len(s)))), len(s))
    thr_r = float(s[k - 1])
    patch_recall = float((np.array(f_scores) >= thr_r).mean())
    patch_fp = float((np.array(a_scores) >= thr_r).mean())

    return {'pixel_auc': round(pix_auc, 4) if pix_auc else None,
            'pixel_bestF1': round(pix_bestf1, 4),
            'patch_auc': round(patch_auc, 4),
            'patch_fp_at_recall': round(patch_fp, 4),
            'patch_recall': round(patch_recall, 4),
            'n_forged_patches': len(f_scores), 'n_auth_patches': len(a_scores),
            'n_pixels_eval': int(L.size)}


def main():
    if not os.path.exists('reports/forged_flags.json'):
        raise SystemExit("Run diagnostics/cache_forged_flags.py first.")
    print(f"Device: {DEVICE} | seed={SEED} steps={STEPS} patch={PATCH} amp={USE_AMP} "
          f"regimes={REGIMES}", flush=True)
    (tr_f, tr_a), (te_f, te_a) = load_rtm()
    print(f"RTM train: forged={len(tr_f)} auth={len(tr_a)} | test: forged={len(te_f)} auth={len(te_a)}",
          flush=True)

    # fixed eval patch set (native crops), reused identically across regimes
    fpatch, apatch = build_eval_patches(te_f, te_a, random.Random(SEED + 777))
    print(f"Eval patches: forged={len(fpatch)} auth={len(apatch)} (native {PATCH}px, shared across regimes)",
          flush=True)

    results = {'config': {'seed': SEED, 'steps': STEPS, 'patch': PATCH, 'micro_batch': MICRO_BATCH,
                          'accum': ACCUM, 'lr': LR, 'pos_weight': POS_WEIGHT,
                          'p_forged': P_FORGED, 'p_hardneg': P_HARDNEG,
                          'target_recall': TARGET_RECALL, 'amp': USE_AMP, 'ckpt': CKPT,
                          'note': 'r256 = same native crop downsampled to 256; isolates resolution.'},
               'regimes': {}}
    for regime in REGIMES:
        print(f"\n===== REGIME: {regime} =====", flush=True)
        model = train_regime(regime, tr_f, tr_a)
        entry = eval_regime(model, regime, fpatch, apatch, random.Random(SEED + 999))
        results['regimes'][regime] = entry
        print(f"  {regime}: pixelAUC={entry['pixel_auc']} pixelF1={entry['pixel_bestF1']} | "
              f"patchAUC={entry['patch_auc']} patchFP@rec{int(TARGET_RECALL*100)}="
              f"{entry['patch_fp_at_recall']} (rec={entry['patch_recall']})", flush=True)
        del model; torch.cuda.empty_cache()

    print("\n===== SUMMARY: r512 vs r256 (same crops; isolates resolution) =====", flush=True)
    if 'r256' in results['regimes'] and 'r512' in results['regimes']:
        b = results['regimes']['r256']; p = results['regimes']['r512']
        for key in ('pixel_auc', 'pixel_bestF1', 'patch_auc'):
            if b[key] is not None and p[key] is not None:
                print(f"  {key}: {b[key]} -> {p[key]} ({p[key]-b[key]:+.3f})", flush=True)
        print(f"  patch_fp_at_recall: {b['patch_fp_at_recall']} -> {p['patch_fp_at_recall']} "
              f"({p['patch_fp_at_recall']-b['patch_fp_at_recall']:+.3f}) (lower better)", flush=True)

    os.makedirs('reports', exist_ok=True)
    out = f"reports/highres_patch_rtm_v2_seed{SEED}.json"
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == '__main__':
    main()
