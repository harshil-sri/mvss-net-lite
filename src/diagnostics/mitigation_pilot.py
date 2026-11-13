"""
Paper 1 — Mitigation pilot (Week 3-4 centerpiece; feeds the go/no-go).

Question: does breaking the domain->label correlation recover WITHIN-domain
tampered-vs-authentic discrimination where ERM (and adversarial DANN) cannot?

Two-domain controlled setup (RTM + CASIAv2 — the two domains that contain BOTH
forged and authentic images, so 4 groups exist):
    groups = {RTM_forged, RTM_auth, CASIA_forged, CASIA_auth}

Everything held constant across methods: ImageNet init (same seed), same optimizer
step budget, same batch size, same LR, same FIXED seg pos_weight (=100, matching the
base-rate experiment), same fixed held-out eval (from TEST split).

Only the TRAINING STRATEGY varies:
  - erm      : natural sampling of pooled data (domain-label correlation PRESENT).
  - balanced : equal-probability group sampling -> P(forged|domain)=0.5 for BOTH
               domains -> domain carries no label info (kills the shortcut's usefulness).
  - groupdro : natural sampling + online worst-group loss reweighting (Sagawa et al.).
  - dann     : natural sampling + adversarial domain-invariance head (the failed baseline).

Metrics on the fixed eval set (per domain):
  - authentic-FP, forged-recall (image-level fire@0.5) -> discrimination gap (rec-FP)
  - worst-domain discrimination (the number the fix must raise)
  - RTM-FP vs CASIA-FP disparity (does the fix equalize the domain-locked prior?)
  - pixel IoU on forged images
  - f4 domain-probe accuracy (did the method reduce linear domain separability?)

Read-only w.r.t. existing checkpoints/manifest. Writes reports/mitigation_pilot.json.
Requires reports/forged_flags.json (diagnostics/cache_forged_flags.py).

Run in tmux:
  PYTHONPATH=/home/harshil/mvss-net-lite \
  /home/harshil/.venvs/global/bin/python3 diagnostics/mitigation_pilot.py
"""
import os, json, random, math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt'
DOMAINS = ['RTM', 'CASIAv2']
DOM_ID = {'RTM': 0, 'CASIAv2': 1}
GROUPS = ['RTM_forged', 'RTM_auth', 'CASIAv2_forged', 'CASIAv2_auth']
GID = {g: i for i, g in enumerate(GROUPS)}

STEPS = int(os.environ.get('PILOT_STEPS', 1200))
BATCH = 8
LR = 1e-4
POS_WEIGHT = float(os.environ.get('PILOT_POSW', 20.0))
DRO_ETA = 0.01
DECORR_LAMBDA = float(os.environ.get('PILOT_DECORR', 10.0))
LOGITADJ_TAU = float(os.environ.get('PILOT_TAU', 1.0))
INIT = os.environ.get('PILOT_INIT', 'imagenet')   # 'imagenet' or 'run4'
SEED = int(os.environ.get('PILOT_SEED', 42))
N_EVAL_PER_GROUP = int(os.environ.get('PILOT_EVAL', 250))   # cap per (domain,label) eval cell
METHODS = os.environ.get('PILOT_METHODS', 'erm,balanced,groupdro,dann,decorr').split(',')


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def group_name(domain, forged):
    return f"{domain}_{'forged' if forged else 'auth'}"


def load_pools():
    manifest = json.load(open('reports/manifest.json'))
    flags = json.load(open('reports/forged_flags.json'))
    rng = random.Random(SEED)

    # train pools by group
    train_by_group = defaultdict(list)   # gid -> [(img,mask,domain)]
    for s in manifest['train']:
        d = s['dataset']
        if d not in DOMAINS:
            continue
        f = flags['train'].get(s['image'])
        if f is None:
            continue
        train_by_group[GID[group_name(d, f)]].append((s['image'], s['mask'], d))

    # fixed eval cells by group (from TEST split)
    eval_by_group = defaultdict(list)
    for s in manifest['test']:
        d = s['dataset']
        if d not in DOMAINS:
            continue
        f = flags['test'].get(s['image'])
        if f is None:
            continue
        eval_by_group[GID[group_name(d, f)]].append((s['image'], s['mask'], d))
    for g in eval_by_group:
        rng.shuffle(eval_by_group[g])
        eval_by_group[g] = eval_by_group[g][:N_EVAL_PER_GROUP]
    return train_by_group, eval_by_group


def build_flat(train_by_group):
    """Flat arrays over all train samples: pairs, group_id, domain_id."""
    pairs, gids, dids = [], [], []
    for g, lst in train_by_group.items():
        for (img, mask, d) in lst:
            pairs.append((img, mask)); gids.append(g); dids.append(DOM_ID[d])
    return pairs, np.array(gids), np.array(dids)


def make_batches(gids, method, rng):
    """Precompute STEPS batches of flat indices per sampling policy."""
    n = len(gids)
    idx_by_group = defaultdict(list)
    for i, g in enumerate(gids):
        idx_by_group[g].append(i)
    present_groups = sorted(idx_by_group.keys())
    batches = []
    for _ in range(STEPS):
        if method == 'balanced':
            b = []
            for _k in range(BATCH):
                g = rng.choice(present_groups)
                b.append(rng.choice(idx_by_group[g]))
        else:  # erm / groupdro / dann -> natural sampling
            b = [rng.randrange(n) for _k in range(BATCH)]
        batches.append(b)
    return batches


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
    bce = bce.flatten(1).mean(dim=1)                    # (B,)
    probs = torch.sigmoid(logits).flatten(1)
    t = targets.flatten(1)
    inter = (probs * t).sum(dim=1)
    region = 1 - (2 * inter + 1e-6) / (probs.sum(dim=1) + t.sum(dim=1) + 1e-6)
    return bce + region                                  # (B,)


def decorrelation_loss(f4_gap, domain_ids):
    """Penalize linear predictability of domain from f4 features (targets the ~0.99
    linear domain-probe). cov between each centered feature and centered domain label."""
    d = domain_ids.float()
    if d.std() < 1e-6:   # single domain in this batch -> no signal
        return torch.zeros((), device=f4_gap.device)
    d = (d - d.mean())
    f = f4_gap - f4_gap.mean(dim=0, keepdim=True)
    cov = (f * d.unsqueeze(1)).mean(dim=0)     # (C,)
    return (cov ** 2).mean()


# ---- logit adjustment (Menon et al.) for the domain base-rate shortcut ----
# per-domain-id -> training logit bias tau*logit(pixel_forged_prior). Set in main().
LOGITADJ_BIAS = {}


def estimate_domain_pixel_prior(train_by_group):
    """Effective forged-PIXEL prior per domain at the training resolution (256, post
    adaptive-max-pool), averaged over ALL images incl. authentic. Captures BOTH the
    image base-rate AND forgery size (the two confounds we diagnosed)."""
    rng = random.Random(SEED)
    prior = {}
    for d in DOMAINS:
        pairs = []
        for g in (GID[f'{d}_forged'], GID[f'{d}_auth']):
            pairs += [(a, b) for (a, b, _dm) in train_by_group.get(g, [])]
        rng.shuffle(pairs)
        pairs = pairs[:300]
        ds = ForgeryDataset(pairs, is_train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=3)
        tot, npix = 0.0, 0
        for imgs, masks, edges in loader:
            tot += float((masks > 0.5).float().sum()); npix += masks.numel()
        prior[DOM_ID[d]] = tot / max(1, npix)
    return prior


def train_method(method, pairs, gids, dids):
    set_seed(SEED)
    rng = random.Random(SEED)
    model = MVSSNetLite().to(DEVICE)   # ImageNet init, same seed
    if INIT == 'run4':
        ck = torch.load(CKPT, map_location=DEVICE)
        model.load_state_dict(ck.get('model_state_dict', ck.get('model', ck)), strict=False)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    batches = make_batches(gids, method, rng)
    ds = IdxDataset(pairs)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=batches, num_workers=3)

    # f4 capture for decorrelation
    f4_store = {}
    hook = None
    if method == 'decorr':
        hook = model.backbone.fuse4.register_forward_hook(
            lambda m, i, o: f4_store.__setitem__('f4', o))

    q = torch.ones(len(GROUPS), device=DEVICE) / len(GROUPS)   # groupdro weights
    model.train()
    step = 0
    running = 0.0
    for imgs, masks, idxs in loader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        idxs = idxs.numpy()
        opt.zero_grad()

        lambda_p = None
        if method == 'dann':
            p = step / float(STEPS)
            lambda_p = (2.0 / (1.0 + math.exp(-10.0 * p))) - 1.0
            seg, edge, dom_logits = model(imgs, lambda_p=lambda_p)
        else:
            seg, edge = model(imgs)

        # logit adjustment: train on prior-shifted logits so RAW logits (used at test)
        # become prior-free. bias = tau*logit(pixel_forged_prior_of_domain).
        seg_for_loss = seg
        if method == 'logitadj':
            bias = torch.tensor([LOGITADJ_BIAS[dids[i]] for i in idxs],
                                device=DEVICE).view(-1, 1, 1, 1)
            seg_for_loss = seg + bias

        ps_loss = per_sample_seg_loss(seg_for_loss, masks, POS_WEIGHT)   # (B,)

        if method == 'groupdro':
            batch_g = torch.tensor([gids[i] for i in idxs], device=DEVICE)
            loss = torch.zeros((), device=DEVICE)
            present = torch.unique(batch_g)
            group_losses = {}
            for g in present:
                gl = ps_loss[batch_g == g].mean()
                group_losses[int(g)] = gl
                q[g] = q[g] * torch.exp(DRO_ETA * gl.detach())
            q = q / q.sum()
            for g_int, gl in group_losses.items():
                loss = loss + q[g_int] * gl
        else:
            loss = ps_loss.mean()

        if method == 'dann':
            dom_t = torch.tensor([dids[i] for i in idxs], device=DEVICE, dtype=torch.long)
            dom_loss = F.cross_entropy(dom_logits, dom_t)
            loss = loss + 0.1 * dom_loss

        if method == 'decorr':
            f4_gap = f4_store['f4'].flatten(2).mean(dim=2)   # (B,C)
            dom_t = torch.tensor([dids[i] for i in idxs], device=DEVICE)
            loss = loss + DECORR_LAMBDA * decorrelation_loss(f4_gap, dom_t)

        loss.backward()
        opt.step()
        model.backbone.noise_extractor.normalize_weights()
        running += float(loss.detach()); step += 1
        if step % 200 == 0:
            print(f"    [{method}] step {step}/{STEPS} loss={running/200:.4f}")
            running = 0.0
    if hook is not None:
        hook.remove()
    return model


@torch.no_grad()
def evaluate(model, eval_by_group):
    model.eval()
    out = {}
    feats_all, dom_all = [], []
    # image-level scores per domain for threshold-free AUC
    scores = {d: {'forged': [], 'auth': []} for d in DOMAINS}
    for g in range(len(GROUPS)):
        pairs3 = eval_by_group.get(g, [])
        if not pairs3:
            continue
        domain = GROUPS[g].rsplit('_', 1)[0]
        forged = GROUPS[g].endswith('forged')
        pairs = [(a, b) for (a, b, d) in pairs3]
        ds = ForgeryDataset(pairs, is_train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=3)

        fired_n, n = 0, 0
        iou_sum, iou_n = 0.0, 0
        feats = {}
        h = model.backbone.fuse4.register_forward_hook(
            lambda m, i, o: feats.__setitem__('f4', o.detach()))
        for imgs, masks, edges in loader:
            imgs = imgs.to(DEVICE); masks = masks.to(DEVICE)
            seg, edge = model(imgs)
            prob = torch.sigmoid(seg)
            # image-level score = mean probability (threshold-free discrimination signal)
            img_score = prob.flatten(1).mean(dim=1).cpu().numpy()
            scores[domain]['forged' if forged else 'auth'].extend(img_score.tolist())
            pred = (prob > 0.5)
            fired = pred.flatten(1).any(dim=1)
            fired_n += int(fired.sum().item()); n += imgs.shape[0]
            if forged:
                p = pred.flatten(1).float(); t = (masks.flatten(1) > 0.5).float()
                inter = (p * t).sum(dim=1); union = ((p + t) > 0).float().sum(dim=1)
                valid = union > 0
                if valid.any():
                    iou_sum += float((inter[valid] / union[valid]).sum()); iou_n += int(valid.sum())
            gap = feats['f4'].flatten(2).mean(dim=2).cpu().numpy()
            feats_all.append(gap); dom_all += [DOM_ID[domain]] * imgs.shape[0]
        h.remove()

        fire_rate = fired_n / max(1, n)
        cell = {'n': n, 'fire_rate': round(fire_rate, 4)}
        if forged:
            cell['pixel_iou'] = round(iou_sum / max(1, iou_n), 4)
        out[GROUPS[g]] = cell

    # per-domain discrimination: threshold-free AUC (forged vs auth image scores)
    per_dom = {}
    for d in DOMAINS:
        fp = out.get(f"{d}_auth", {}).get('fire_rate')
        rec = out.get(f"{d}_forged", {}).get('fire_rate')
        gap = None if (fp is None or rec is None) else round(rec - fp, 4)
        f_sc, a_sc = scores[d]['forged'], scores[d]['auth']
        auc = None
        if f_sc and a_sc:
            ys = [1] * len(f_sc) + [0] * len(a_sc)
            auc = round(float(roc_auc_score(ys, f_sc + a_sc)), 4)
        per_dom[d] = {'auc': auc, 'auth_fp': fp, 'forged_recall': rec,
                      'fire_gap': gap,
                      'forged_pixel_iou': out.get(f"{d}_forged", {}).get('pixel_iou')}
    aucs = [v['auc'] for v in per_dom.values() if v['auc'] is not None]
    fps = [v['auth_fp'] for v in per_dom.values() if v['auth_fp'] is not None]

    X = np.concatenate(feats_all); y = np.array(dom_all)
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    dom_acc = float(cross_val_score(pipe, X, y, cv=skf, scoring='accuracy').mean())

    return {
        'groups': out,
        'per_domain': per_dom,
        'worst_domain_auc': None if not aucs else round(min(aucs), 4),
        'mean_domain_auc': None if not aucs else round(float(np.mean(aucs)), 4),
        'fp_disparity': None if len(fps) < 2 else round(float(max(fps) - min(fps)), 4),
        'f4_domain_probe_acc': round(dom_acc, 4),
    }


def main():
    if not os.path.exists('reports/forged_flags.json'):
        raise SystemExit("Run diagnostics/cache_forged_flags.py first.")
    print(f"Device: {DEVICE} | STEPS={STEPS} batch={BATCH} lr={LR} pos_weight={POS_WEIGHT} (FIXED)")
    train_by_group, eval_by_group = load_pools()
    pairs, gids, dids = build_flat(train_by_group)
    print("Train group sizes:", {GROUPS[g]: len(train_by_group[g]) for g in sorted(train_by_group)})
    print("Eval  group sizes:", {GROUPS[g]: len(eval_by_group[g]) for g in sorted(eval_by_group)})
    # natural base-rate per domain in train (context)
    for d in DOMAINS:
        f = len(train_by_group[GID[f'{d}_forged']]); a = len(train_by_group[GID[f'{d}_auth']])
        print(f"  {d} natural forged base-rate = {f/max(1,f+a):.3f}  ({f}f/{a}a)")

    # logit-adjustment bias per domain (only computed if needed)
    global LOGITADJ_BIAS
    if 'logitadj' in METHODS:
        prior = estimate_domain_pixel_prior(train_by_group)
        for did, pi in prior.items():
            pi_c = min(max(pi, 1e-4), 0.5)
            LOGITADJ_BIAS[did] = LOGITADJ_TAU * math.log(pi_c / (1 - pi_c))
        print(f"  logitadj pixel-priors={{{', '.join(f'{DOMAINS[d]}:{prior[DOM_ID[DOMAINS[d]]]:.4f}' for d in range(len(DOMAINS)))}}} "
              f"tau={LOGITADJ_TAU} bias={{{', '.join(f'{DOMAINS[d]}:{LOGITADJ_BIAS[DOM_ID[DOMAINS[d]]]:.2f}' for d in range(len(DOMAINS)))}}}")

    results = {'config': {'domains': DOMAINS, 'steps': STEPS, 'batch': BATCH, 'lr': LR,
                          'pos_weight_fixed': POS_WEIGHT, 'dro_eta': DRO_ETA, 'seed': SEED,
                          'init': INIT, 'decorr_lambda': DECORR_LAMBDA, 'logitadj_tau': LOGITADJ_TAU,
                          'metric': 'per-domain AUC of image-mean-prob (threshold-free)'},
               'methods': {}}

    for method in METHODS:
        print(f"\n===== method: {method} =====")
        model = train_method(method, pairs, gids, dids)
        ev = evaluate(model, eval_by_group)
        results['methods'][method] = ev
        pd = ev['per_domain']
        print(f"  per-domain: " + " | ".join(
            f"{d}: AUC={pd[d]['auc']} FP={pd[d]['auth_fp']} rec={pd[d]['forged_recall']}"
            for d in DOMAINS))
        print(f"  worst-AUC={ev['worst_domain_auc']} mean-AUC={ev['mean_domain_auc']} "
              f"FP-disparity={ev['fp_disparity']} f4-domain-acc={ev['f4_domain_probe_acc']}")
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    os.makedirs('reports', exist_ok=True)
    outpath = f"reports/mitigation_pilot_{INIT}_seed{SEED}.json"
    json.dump(results, open(outpath, 'w'), indent=2)
    print(f"\nSaved {outpath}")

    # summary table (AUC = threshold-free within-domain discrimination; 0.5=chance)
    print("\n===== SUMMARY (higher worst-AUC = better; lower FP-disparity = better) =====")
    print(f"{'method':>10s} | worst-AUC | mean-AUC | FP-disp | f4-dom-acc | "
          f"RTM AUC/FP | CASIA AUC/FP")
    for m in METHODS:
        e = results['methods'][m]; pd = e['per_domain']
        print(f"{m:>10s} | {str(e['worst_domain_auc']):>9s} | "
              f"{str(e['mean_domain_auc']):>8s} | {str(e['fp_disparity']):>7s} | "
              f"{str(e['f4_domain_probe_acc']):>10s} | "
              f"{pd['RTM']['auc']}/{pd['RTM']['auth_fp']} | "
              f"{pd['CASIAv2']['auc']}/{pd['CASIAv2']['auth_fp']}")

    # verdict vs ERM
    erm = results['methods'].get('erm', {}).get('worst_domain_auc')
    print(f"\nERM worst-domain AUC = {erm}  (0.5 = no within-domain discrimination)")
    for m in METHODS:
        if m == 'erm':
            continue
        w = results['methods'][m]['worst_domain_auc']
        if erm is not None and w is not None:
            delta = round(w - erm, 4)
            tag = "IMPROVES" if delta > 0.02 else ("~same" if abs(delta) <= 0.02 else "WORSE")
            print(f"  {m}: worst-AUC {w}  (Δ vs ERM {delta:+}) -> {tag}")


if __name__ == '__main__':
    main()
