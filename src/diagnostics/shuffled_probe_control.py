import os, json, random
from collections import defaultdict, Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CKPT = 'model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt'
DOMAINS = ['RTM', 'MIDV500', 'CASIAv2', 'DEFACTO']
N_PER_DOMAIN = int(os.environ.get('SP_N_PER_DOMAIN', 250))
BATCH = 8
SEED = int(os.environ.get('SP_SEED', 42))


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load_model(init='run4'):
    model = MVSSNetLite().to(DEVICE)
    if init == 'run4':
        ckpt = torch.load(CKPT, map_location=DEVICE)
        state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
        model.load_state_dict(state, strict=False)
    elif init == 'random':
        pass
    else:
        raise ValueError(f"Unknown init: {init}")
    model.eval()
    return model


def sample_pairs():
    manifest = json.load(open('reports/manifest.json'))
    by_dom = defaultdict(list)
    for s in manifest['test']:
        if s['dataset'] in DOMAINS:
            by_dom[s['dataset']].append((s['image'], s['mask']))
    rng = random.Random(SEED)
    pairs, dom_labels = [], []
    for d in DOMAINS:
        lst = list(by_dom[d])
        rng.shuffle(lst)
        chosen = lst[:N_PER_DOMAIN]
        pairs.extend(chosen)
        dom_labels.extend([d] * len(chosen))
    return pairs, dom_labels


def gap(t):
    return t.flatten(2).mean(dim=2)


@torch.no_grad()
def extract_f4_features(model, pairs):
    feats = {}
    h = model.backbone.fuse4.register_forward_hook(lambda m, i, o: feats.__setitem__('f4', o.detach()))
    ds = ForgeryDataset(pairs, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)

    F4 = []
    for imgs, masks, edges in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        seg, edge = model(imgs)
        F4.append(gap(feats['f4']).cpu().numpy())
    h.remove()
    return np.concatenate(F4, axis=0)


def evaluate_probe(X, y_labels, seed):
    Xtr, Xte, ytr, yte = train_test_split(X, y_labels, test_size=0.3, stratify=y_labels, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = accuracy_score(yte, pred)
    cm = confusion_matrix(yte, pred, labels=DOMAINS)
    return {
        'accuracy': round(float(acc), 4),
        'confusion': cm.tolist()
    }


def main():
    if not os.path.exists(CKPT):
        raise SystemExit(f"Checkpoint not found: {CKPT}")
    set_seed(SEED)
    print(f"Device: {DEVICE} | CKPT: {CKPT} | SEED: {SEED}", flush=True)

    pairs, dom_labels = sample_pairs()
    print("Sampled per domain:", dict(Counter(dom_labels)), flush=True)
    y_true = np.array(dom_labels)

    rng_shuff = np.random.RandomState(SEED)
    y_shuffled = y_true.copy()
    rng_shuff.shuffle(y_shuffled)

    results = {
        'config': {
            'seed': SEED,
            'n_per_domain': N_PER_DOMAIN,
            'domains': DOMAINS,
            'chance': round(1.0 / len(DOMAINS), 4)
        },
        'probes': {}
    }

    print("\nExtracting f4 features for random init...", flush=True)
    set_seed(SEED)
    model_rnd = load_model(init='random')
    f4_rnd = extract_f4_features(model_rnd, pairs)

    print("Extracting f4 features for run4 checkpoint...", flush=True)
    model_run4 = load_model(init='run4')
    f4_run4 = extract_f4_features(model_run4, pairs)

    for init_name, X in [('random_init', f4_rnd), ('run4_checkpoint', f4_run4)]:
        res_true = evaluate_probe(X, y_true, seed=SEED)
        res_shuff = evaluate_probe(X, y_shuffled, seed=SEED)
        results['probes'][init_name] = {
            'true_labels': res_true,
            'shuffled_labels': res_shuff
        }
        print(f"[{init_name}] true_acc={res_true['accuracy']:.4f} | shuffled_acc={res_shuff['accuracy']:.4f} (chance={results['config']['chance']:.4f})", flush=True)

    os.makedirs('reports', exist_ok=True)
    out_file = f"reports/shuffled_probe_seed{SEED}.json"
    json.dump(results, open(out_file, 'w'), indent=2)
    print(f"\nSaved {out_file}", flush=True)


if __name__ == '__main__':
    main()
