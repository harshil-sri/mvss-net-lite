import os, sys, json, random, math
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
POSSIBLE_CKPTS = [
    'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt',
    'src/model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt',
    os.path.join(os.path.dirname(__file__), '..', 'model', 'checkpoints', 'stage2run4_mvss_lite_ep8_bestval.pt')
]
CKPT = next((p for p in POSSIBLE_CKPTS if os.path.exists(p)), None)

DOMAINS = ['RTM', 'CASIAv2']
SEED = int(os.environ.get('CB_SEED', 42))
N_EVAL_PER_CELL = int(os.environ.get('CB_EVAL', 300))
TARGET_RECALL = float(os.environ.get('CB_RECALL', 0.85))
CALIB_FRAC = float(os.environ.get('CB_CALIB_FRAC', 0.5))
BATCH = 8


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_pools():
    if not os.path.exists('reports/manifest.json') or not os.path.exists('reports/forged_flags.json'):
        raise SystemExit("Missing reports/manifest.json or reports/forged_flags.json.")
    manifest = json.load(open('reports/manifest.json'))
    flags_all = json.load(open('reports/forged_flags.json'))
    rng = random.Random(SEED)

    cells = defaultdict(list)
    for s in manifest['test']:
        d = s['dataset']
        if d not in DOMAINS:
            continue
        f = flags_all['test'].get(s['image'])
        if f is None:
            continue
        cells[(d, bool(f))].append((s['image'], s['mask']))

    calib_cells, test_cells = {}, {}
    for key, lst in cells.items():
        rng.shuffle(lst)
        lst = lst[:N_EVAL_PER_CELL]
        n_calib = int(len(lst) * CALIB_FRAC)
        calib_cells[key] = lst[:n_calib]
        test_cells[key] = lst[n_calib:]
    return calib_cells, test_cells


def build_model():
    if CKPT is None or not os.path.exists(CKPT):
        raise SystemExit(
            f"Checkpoint not found. Tried paths:\n" + "\n".join(f"  - {p}" for p in POSSIBLE_CKPTS)
        )
    model = MVSSNetLite().to(DEVICE)
    ck = torch.load(CKPT, map_location=DEVICE)
    model.load_state_dict(ck.get('model_state_dict', ck.get('model', ck)), strict=False)
    return model


@torch.no_grad()
def collect_scores(model, cells):
    model.eval()
    out = defaultdict(lambda: {'segmean': [], 'segtopk': []})
    for key, pairs in cells.items():
        if not pairs:
            continue
        ds = ForgeryDataset(pairs, is_train=False)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)
        for imgs, masks, edges in loader:
            imgs = imgs.to(DEVICE)
            seg, edge = model(imgs)
            prob = torch.sigmoid(seg).flatten(1)
            segmean = prob.mean(dim=1).cpu().tolist()
            k = max(1, int(0.01 * prob.shape[1]))
            segtopk = prob.topk(k, dim=1).values.mean(dim=1).cpu().tolist()
            out[key]['segmean'] += segmean
            out[key]['segtopk'] += segtopk
    return out


def pool_data(cells_scores, readout):
    scores, labels = [], []
    for (dom, forged), val in cells_scores.items():
        s = val[readout]
        y = 1 if forged else 0
        scores.extend(s)
        labels.extend([y] * len(s))
    return np.array(scores, dtype=np.float64), np.array(labels, dtype=np.int32)


def compute_ece(probs, labels, n_bins=15):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(probs) == 0:
        return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        low, high = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (probs >= low) & (probs <= high)
        else:
            mask = (probs >= low) & (probs < high)
        count = np.sum(mask)
        if count > 0:
            bin_acc = np.mean(labels[mask])
            bin_conf = np.mean(probs[mask])
            ece += (count / n) * np.abs(bin_acc - bin_conf)
    return float(ece)


def fit_temperature(scores, labels):
    eps = 1e-7
    clipped = np.clip(scores, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))
    t_candidates = np.logspace(np.log10(0.2), np.log10(5.0), num=100)
    best_t = 1.0
    best_nll = float('inf')
    for t in t_candidates:
        scaled_logits = logits / t
        scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        loss = log_loss(labels, scaled_probs)
        if loss < best_nll:
            best_nll = loss
            best_t = float(t)
    return best_t


def apply_temperature(scores, t):
    eps = 1e-7
    clipped = np.clip(scores, eps, 1.0 - eps)
    logits = np.log(clipped / (1.0 - clipped))
    scaled_logits = logits / t
    return 1.0 / (1.0 + np.exp(-scaled_logits))


def threshold_at_recall(scores, target_recall):
    s = np.sort(np.asarray(scores))[::-1]
    k = max(1, int(math.ceil(target_recall * len(s))))
    k = min(k, len(s))
    return float(s[k - 1])


def evaluate_split(cells_scores, readout, calib_cells_scores=None):
    if calib_cells_scores is not None:
        pooled_f = []
        for d in DOMAINS:
            pooled_f.extend(calib_cells_scores.get((d, True), {}).get(readout, []))
        thr = threshold_at_recall(pooled_f, TARGET_RECALL)
    else:
        pooled_f = []
        for d in DOMAINS:
            pooled_f.extend(cells_scores.get((d, True), {}).get(readout, []))
        thr = threshold_at_recall(pooled_f, TARGET_RECALL)

    metrics = {}
    for d in DOMAINS:
        f = np.array(cells_scores.get((d, True), {}).get(readout, []))
        a = np.array(cells_scores.get((d, False), {}).get(readout, []))
        rec = float((f >= thr).mean()) if len(f) else None
        fp = float((a >= thr).mean()) if len(a) else None
        all_s = np.concatenate([f, a]) if len(f) and len(a) else np.array([])
        all_y = np.concatenate([np.ones(len(f)), np.zeros(len(a))]) if len(f) and len(a) else np.array([])
        ece = compute_ece(all_s, all_y) if len(all_s) else None
        auc = float(roc_auc_score(all_y, all_s)) if len(all_s) and len(np.unique(all_y)) > 1 else None
        metrics[d] = {
            'threshold': round(thr, 4),
            'recall': round(rec, 4) if rec is not None else None,
            'auth_fp': round(fp, 4) if fp is not None else None,
            'ece': round(ece, 4) if ece is not None else None,
            'auc': round(auc, 4) if auc is not None else None
        }
    return metrics


def main():
    set_seed(SEED)
    print(f"Device: {DEVICE} | CKPT: {CKPT} | seed: {SEED}", flush=True)
    calib_cells, test_cells = load_pools()
    model = build_model()

    calib_scores = collect_scores(model, calib_cells)
    test_scores = collect_scores(model, test_cells)

    readouts = ['segmean', 'segtopk']
    results = {
        'config': {
            'domains': DOMAINS,
            'seed': SEED,
            'target_recall': TARGET_RECALL,
            'calib_frac': CALIB_FRAC,
            'ckpt': CKPT,
            'readouts': readouts
        },
        'per_readout': {}
    }

    raw_samples = {}
    for readout in readouts:
        calib_s, calib_y = pool_data(calib_scores, readout)
        t_opt = fit_temperature(calib_s, calib_y)

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        iso.fit(calib_s, calib_y)

        uncal_calib_scores = {k: {readout: v[readout]} for k, v in calib_scores.items()}
        temp_calib_scores = {
            k: {readout: apply_temperature(np.array(v[readout]), t_opt).tolist()}
            for k, v in calib_scores.items()
        }
        iso_calib_scores = {
            k: {readout: iso.predict(np.array(v[readout])).tolist()}
            for k, v in calib_scores.items()
        }

        uncal_test_scores = {k: {readout: v[readout]} for k, v in test_scores.items()}
        temp_test_scores = {
            k: {readout: apply_temperature(np.array(v[readout]), t_opt).tolist()}
            for k, v in test_scores.items()
        }
        iso_test_scores = {
            k: {readout: iso.predict(np.array(v[readout])).tolist()}
            for k, v in test_scores.items()
        }

        uncal_metrics_fixed = evaluate_split(uncal_test_scores, readout)
        temp_metrics_fixed = evaluate_split(temp_test_scores, readout)
        iso_metrics_fixed = evaluate_split(iso_test_scores, readout)

        uncal_metrics_refit = evaluate_split(uncal_test_scores, readout, calib_cells_scores=uncal_calib_scores)
        temp_metrics_refit = evaluate_split(temp_test_scores, readout, calib_cells_scores=temp_calib_scores)
        iso_metrics_refit = evaluate_split(iso_test_scores, readout, calib_cells_scores=iso_calib_scores)

        results['per_readout'][readout] = {
            'optimal_temperature': round(t_opt, 4),
            'uncalibrated': uncal_metrics_fixed,
            'temperature_calibrated': temp_metrics_fixed,
            'isotonic_calibrated': iso_metrics_fixed,
            'uncalibrated_refit': uncal_metrics_refit,
            'temperature_calibrated_refit': temp_metrics_refit,
            'isotonic_calibrated_refit': iso_metrics_refit,
            'fixed_threshold': {
                'uncalibrated': uncal_metrics_fixed,
                'temperature_calibrated': temp_metrics_fixed,
                'isotonic_calibrated': iso_metrics_fixed
            },
            'matched_recall_threshold_refit': {
                'uncalibrated': uncal_metrics_refit,
                'temperature_calibrated': temp_metrics_refit,
                'isotonic_calibrated': iso_metrics_refit
            }
        }

        raw_samples[readout] = {
            'calib': {
                f"{d}_{'forged' if forged else 'auth'}": {
                    'uncalibrated': [round(float(s), 6) for s in uncal_calib_scores.get((d, forged), {}).get(readout, [])],
                    'temperature_calibrated': [round(float(s), 6) for s in temp_calib_scores.get((d, forged), {}).get(readout, [])],
                    'isotonic_calibrated': [round(float(s), 6) for s in iso_calib_scores.get((d, forged), {}).get(readout, [])],
                    'labels': [1 if forged else 0] * len(uncal_calib_scores.get((d, forged), {}).get(readout, []))
                }
                for d in DOMAINS for forged in [True, False]
            },
            'test': {
                f"{d}_{'forged' if forged else 'auth'}": {
                    'uncalibrated': [round(float(s), 6) for s in uncal_test_scores.get((d, forged), {}).get(readout, [])],
                    'temperature_calibrated': [round(float(s), 6) for s in temp_test_scores.get((d, forged), {}).get(readout, [])],
                    'isotonic_calibrated': [round(float(s), 6) for s in iso_test_scores.get((d, forged), {}).get(readout, [])],
                    'labels': [1 if forged else 0] * len(uncal_test_scores.get((d, forged), {}).get(readout, []))
                }
                for d in DOMAINS for forged in [True, False]
            }
        }

    results['per_sample_scores'] = raw_samples

    print(f"\n===== CALIBRATION BASELINE RESULTS (calib_frac={CALIB_FRAC}) =====", flush=True)
    for readout in readouts:
        res = results['per_readout'][readout]
        print(f"\nReadout: {readout} (Optimal T={res['optimal_temperature']}) [calib_frac={CALIB_FRAC}]")
        print(f"{'Domain':>8s} | {'Variant':>24s} | {'Threshold':>10s} | {'Recall':>8s} | {'Auth-FP':>8s} | {'ECE':>8s} | {'AUC':>8s}")
        for d in DOMAINS:
            for variant, mdict in [('Uncalibrated (fixed)', res['uncalibrated'][d]),
                                  ('Temperature (fixed)', res['temperature_calibrated'][d]),
                                  ('Isotonic (fixed)', res['isotonic_calibrated'][d]),
                                  ('Uncalibrated (refit)', res['uncalibrated_refit'][d]),
                                  ('Temperature (refit)', res['temperature_calibrated_refit'][d]),
                                  ('Isotonic (refit)', res['isotonic_calibrated_refit'][d])]:
                print(f"{d:>8s} | {variant:>24s} | {mdict['threshold']:>10.4f} | {mdict['recall']:>8.4f} | {mdict['auth_fp']:>8.4f} | {mdict['ece']:>8.4f} | {mdict['auc']:>8.4f}")

    os.makedirs('reports', exist_ok=True)
    out_path = f"reports/calibration_baseline_seed{SEED}.json"
    json.dump(results, open(out_path, 'w'), indent=2)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == '__main__':
    main()
