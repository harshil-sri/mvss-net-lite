"""
Paper 1 — Step 2 / ATTEMPT 3: DECISIVE resolution + cropping ablation for RTM.

Motivation (see §4k, §4l): attempt 1 failed (image-max + no hard negatives). Attempt 2 added hard
negatives + pixel readout and recovered localization (pixel-AUC ~0.79) but left two open confounds:
  (i)  its "256" was a 512-crop downsampled to 256, NOT the production full-document 256 resize, so
       it never tested the big lever §4j identified (full-doc downscale destroys tiny forgeries);
  (ii) its pixel-F1 was base-rate-confounded (positives scale with resolution, negatives capped).

v3 settles both with THREE regimes under an IDENTICAL hard-negative training protocol:
  full256 : full document resized to 256x256 (the REAL production pipeline).
  crop256 : 512 native crop downsampled to 256 (crop, low res).
  crop512 : 512 native crop (crop, high res).
Two clean contrasts: CROPPING = full256 vs crop256; RESOLUTION = crop256 vs crop512.

Metrics (base-rate robust / de-confounded):
  PRIMARY   pixel-AUC (threshold-free ranking of forged vs authentic pixels) — base-rate robust.
  pixel-F1  computed at a FIXED pos:neg pixel ratio (1:NEG_RATIO) identical across regimes, so F1 is
            comparable (removes the §4l pooling artifact).
  detection patch/image-level "is this forged" AUC + authentic-FP @ matched recall (segtopk readout).

Saves each regime's checkpoint to model/checkpoints/highres_v3_<regime>_seed<seed>.pt (new files;
Run4 untouched) so metrics can be recomputed later WITHOUT retraining.

Fine-tunes from Run4, RTM-only. 512 fits ~8GB via AMP + grad-accum. Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/highres_patch_rtm_v3.py
"""
import os, json, math, random

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
P_FORGED = float(os.environ.get('HR_PFORGED', 0.5))
P_HARDNEG = float(os.environ.get('HR_PHARDNEG', 0.25))
N_EVAL_FORGED = int(os.environ.get('HR_NEVALF', 200))
N_EVAL_AUTH = int(os.environ.get('HR_NEVALA', 200))
TARGET_RECALL = float(os.environ.get('HR_RECALL', 0.85))
USE_AMP = os.environ.get('HR_AMP', '1') == '1'
REGIMES = os.environ.get('HR_REGIMES', 'full256,crop256,crop512').split(',')
NUM_WORKERS = int(os.environ.get('HR_WORKERS', 1))   # low default: 7.8GB RAM box; 3 workers OOM'd WSL
NEG_RATIO = int(os.environ.get('HR_NEGRATIO', 5))   # negatives per positive pixel (fixed across regimes)
SAVE_CKPT = os.environ.get('HR_SAVE', '1') == '1'


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


def resize_full(img, mask, res):
    img_r = cv2.resize(img, (res, res), interpolation=cv2.INTER_LINEAR)
    mt = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    m_r = F.adaptive_max_pool2d(mt, (res, res)).squeeze().numpy().astype(np.uint8)
    return img_r, m_r


def forged_crop(img, mask, P, rng):
    img, mask = ensure_min_size(img, mask, P)
    H, W = img.shape[:2]
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    j = rng.randrange(len(ys)); cy, cx = int(ys[j]), int(xs[j])
    top = min(max(cy - P // 2, 0), H - P); left = min(max(cx - P // 2, 0), W - P)
    return img[top:top + P, left:left + P], mask[top:top + P, left:left + P]


def empty_crop(img, mask, P, rng, tries=8):
    img, mask = ensure_min_size(img, mask, P)
    H, W = img.shape[:2]
    best = None
    for _ in range(tries):
        top = rng.randint(0, H - P); left = rng.randint(0, W - P)
        cm = mask[top:top + P, left:left + P]
        best = (img[top:top + P, left:left + P], cm)
        if cm.sum() == 0:
            return best
    return best


def to_tensors(img, mask):
    it = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    mt = torch.from_numpy(mask).float().unsqueeze(0)
    return it, mt


def make_train_sample(regime, fp, ap, rng):
    """Return (img, mask) at the regime's input resolution."""
    r = rng.random()
    if regime == 'full256':
        # 50% forged doc / 50% authentic doc, whole document resized to 256
        if r < P_FORGED:
            img, mask = read_img_mask(*fp[rng.randrange(len(fp))])
        else:
            img, mask = read_img_mask(*ap[rng.randrange(len(ap))])
        return resize_full(img, mask, 256)
    # crop regimes: forged-centered / hard-neg / authentic
    for _ in range(6):
        if r < P_FORGED:
            img, mask = read_img_mask(*fp[rng.randrange(len(fp))])
            crop = forged_crop(img, mask, PATCH, rng)
        elif r < P_FORGED + P_HARDNEG:
            img, mask = read_img_mask(*fp[rng.randrange(len(fp))])
            crop = empty_crop(img, mask, PATCH, rng)
        else:
            img, mask = read_img_mask(*ap[rng.randrange(len(ap))])
            crop = empty_crop(img, mask, PATCH, rng)
        if crop is not None:
            break
        r = rng.random()
    ci, cm = crop
    if regime == 'crop256':
        ci, cm = resize_full(ci, cm, 256)
    return ci, cm


class TrainStream(torch.utils.data.Dataset):
    def __init__(self, regime, fp, ap, seed):
        self.regime = regime; self.fp = fp; self.ap = ap; self.seed = seed
    def __len__(self):
        return 10_000_000
    def __getitem__(self, i):
        rng = random.Random(self.seed * 1_000_003 + i)
        img, mask = make_train_sample(self.regime, self.fp, self.ap, rng)
        if rng.random() < 0.5:
            img, mask = img[:, ::-1].copy(), mask[:, ::-1].copy()
        return to_tensors(img, mask)


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


def train_regime(regime, fp, ap):
    set_seed(SEED)
    model = build_model(); model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    ds = TrainStream(regime, fp, ap, SEED)
    total_micro = STEPS * ACCUM
    loader = torch.utils.data.DataLoader(ds, batch_size=MICRO_BATCH, num_workers=NUM_WORKERS,
                                         sampler=range(total_micro * MICRO_BATCH))
    print(f"  [{regime}] training: steps={STEPS} eff_batch={MICRO_BATCH*ACCUM} amp={USE_AMP}", flush=True)
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
    if SAVE_CKPT:
        os.makedirs('model/checkpoints', exist_ok=True)
        path = f"model/checkpoints/highres_v3_{regime}_seed{SEED}.pt"
        torch.save({'model_state_dict': model.state_dict(), 'regime': regime, 'steps': STEPS}, path)
        print(f"    saved checkpoint {path}", flush=True)
    return model


# ------------------------------- eval -------------------------------
def build_eval_sets(regime, forged_test, auth_test, rng):
    """For crop regimes: native 512 forged/authentic crops (resized in-loop for crop256).
       For full256: whole forged/authentic docs (resized to 256 in-loop)."""
    fitems, aitems = [], []
    ftp = forged_test[:]; atp = auth_test[:]
    rng.shuffle(ftp); rng.shuffle(atp)
    if regime == 'full256':
        for pair in ftp:
            if len(fitems) >= N_EVAL_FORGED:
                break
            img, mask = read_img_mask(*pair)
            if mask.sum() > 0:
                fitems.append((img, mask))
        for pair in atp:
            if len(aitems) >= N_EVAL_AUTH:
                break
            img, mask = read_img_mask(*pair)
            aitems.append((img, mask))
        return fitems, aitems
    # crop regimes
    for pair in ftp:
        if len(fitems) >= N_EVAL_FORGED:
            break
        img, mask = read_img_mask(*pair)
        c = forged_crop(img, mask, PATCH, rng)
        if c is not None and c[1].sum() > 0:
            fitems.append(c)
    n_hn = N_EVAL_AUTH // 2
    for pair in ftp:
        if len(aitems) >= n_hn:
            break
        img, mask = read_img_mask(*pair)
        c = empty_crop(img, mask, PATCH, rng)
        if c is not None and c[1].sum() == 0:
            aitems.append(c)
    for pair in atp:
        if len(aitems) >= N_EVAL_AUTH:
            break
        img, mask = read_img_mask(*pair)
        c = empty_crop(img, mask, PATCH, rng)
        if c is not None and c[1].sum() == 0:
            aitems.append(c)
    return fitems, aitems


@torch.no_grad()
def eval_regime(model, regime, fitems, aitems, rng):
    model.eval()
    px_prob, px_lab = [], []
    f_scores, a_scores = [], []

    def infer(img, mask):
        if regime == 'crop256':
            img, mask = resize_full(img, mask, 256)
        elif regime == 'full256':
            img, mask = resize_full(img, mask, 256)
        it = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE) / 255.0
        with torch.amp.autocast('cuda', enabled=USE_AMP):
            seg, _ = model(it)
        return torch.sigmoid(seg).squeeze().float().cpu().numpy(), mask

    for img, mask in fitems:
        prob, m2 = infer(img, mask)
        p = prob.ravel(); m = m2.ravel().astype(np.uint8)
        pos = np.where(m == 1)[0]; neg = np.where(m == 0)[0]
        if len(pos) == 0:
            continue
        n_neg = min(len(neg), NEG_RATIO * len(pos))     # FIXED pos:neg ratio -> comparable F1
        if len(neg) > n_neg:
            neg = np.array(rng.sample(list(neg), n_neg), dtype=int)
        idx = np.concatenate([pos, neg])
        px_prob.append(p[idx]); px_lab.append(m[idx])
        k = max(1, int(0.01 * p.size))
        f_scores.append(float(np.sort(p)[::-1][:k].mean()))
    for img, mask in aitems:
        prob, _ = infer(img, mask)
        p = prob.ravel(); k = max(1, int(0.01 * p.size))
        a_scores.append(float(np.sort(p)[::-1][:k].mean()))

    P = np.concatenate(px_prob); L = np.concatenate(px_lab)
    pix_auc = float(roc_auc_score(L, P)) if L.min() != L.max() else None
    prec, rec, _ = precision_recall_curve(L, P)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    pix_bestf1 = float(np.nanmax(f1))

    ys = [1] * len(f_scores) + [0] * len(a_scores)
    patch_auc = float(roc_auc_score(ys, f_scores + a_scores))
    s = np.sort(np.array(f_scores))[::-1]
    k = min(max(1, int(math.ceil(TARGET_RECALL * len(s)))), len(s))
    thr_r = float(s[k - 1])
    patch_fp = float((np.array(a_scores) >= thr_r).mean())
    patch_rec = float((np.array(f_scores) >= thr_r).mean())

    return {'pixel_auc': round(pix_auc, 4) if pix_auc else None,
            'pixel_bestF1_balanced': round(pix_bestf1, 4),
            'patch_auc': round(patch_auc, 4),
            'patch_fp_at_recall': round(patch_fp, 4),
            'patch_recall': round(patch_rec, 4),
            'neg_ratio': NEG_RATIO,
            'n_forged': len(f_scores), 'n_auth': len(a_scores),
            'n_pixels': int(L.size), 'pos_frac': round(float(L.mean()), 4)}


def main():
    if not os.path.exists('reports/forged_flags.json'):
        raise SystemExit("Run diagnostics/cache_forged_flags.py first.")
    print(f"Device: {DEVICE} | seed={SEED} steps={STEPS} patch={PATCH} amp={USE_AMP} "
          f"regimes={REGIMES} neg_ratio={NEG_RATIO}", flush=True)
    (tr_f, tr_a), (te_f, te_a) = load_rtm()
    print(f"RTM train: forged={len(tr_f)} auth={len(tr_a)} | test: forged={len(te_f)} auth={len(te_a)}",
          flush=True)

    results = {'config': {'seed': SEED, 'steps': STEPS, 'patch': PATCH, 'micro_batch': MICRO_BATCH,
                          'accum': ACCUM, 'lr': LR, 'pos_weight': POS_WEIGHT,
                          'p_forged': P_FORGED, 'p_hardneg': P_HARDNEG, 'neg_ratio': NEG_RATIO,
                          'target_recall': TARGET_RECALL, 'amp': USE_AMP, 'ckpt': CKPT,
                          'regime_defs': {'full256': 'full doc resized 256 (production)',
                                          'crop256': '512 native crop -> 256',
                                          'crop512': '512 native crop'}},
               'regimes': {}}
    for regime in REGIMES:
        print(f"\n===== REGIME: {regime} =====", flush=True)
        model = train_regime(regime, tr_f, tr_a)
        fitems, aitems = build_eval_sets(regime, te_f, te_a, random.Random(SEED + 777))
        print(f"  eval set: forged={len(fitems)} auth={len(aitems)}", flush=True)
        entry = eval_regime(model, regime, fitems, aitems, random.Random(SEED + 999))
        results['regimes'][regime] = entry
        print(f"  {regime}: pixelAUC={entry['pixel_auc']} pixelF1(bal)={entry['pixel_bestF1_balanced']} "
              f"(pos_frac={entry['pos_frac']}) | patchAUC={entry['patch_auc']} "
              f"patchFP@rec{int(TARGET_RECALL*100)}={entry['patch_fp_at_recall']}", flush=True)
        del model; torch.cuda.empty_cache()

    print("\n===== SUMMARY =====", flush=True)
    reg = results['regimes']
    if 'full256' in reg and 'crop256' in reg:
        print("  CROPPING effect (full256 -> crop256, same 256 res):", flush=True)
        for key in ('pixel_auc', 'pixel_bestF1_balanced', 'patch_auc'):
            a, b = reg['full256'][key], reg['crop256'][key]
            if a is not None and b is not None:
                print(f"    {key}: {a} -> {b} ({b-a:+.3f})", flush=True)
    if 'crop256' in reg and 'crop512' in reg:
        print("  RESOLUTION effect (crop256 -> crop512, same crop):", flush=True)
        for key in ('pixel_auc', 'pixel_bestF1_balanced', 'patch_auc'):
            a, b = reg['crop256'][key], reg['crop512'][key]
            if a is not None and b is not None:
                print(f"    {key}: {a} -> {b} ({b-a:+.3f})", flush=True)

    os.makedirs('reports', exist_ok=True)
    out = f"reports/highres_patch_rtm_v3_seed{SEED}.json"
    json.dump(results, open(out, 'w'), indent=2)
    print(f"\nSaved {out}", flush=True)


if __name__ == '__main__':
    main()
