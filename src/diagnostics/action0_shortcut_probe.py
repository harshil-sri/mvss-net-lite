"""
Action 0 — Domain-shortcut probe (Paper 1 foundation).

Question: does the trained forgery model encode DOMAIN (which dataset an image
is from) in its features, rather than tampering? If a linear probe can recover
the dataset label near-perfectly from frozen features, the "domain shortcut"
hypothesis is confirmed.

We probe three feature sites of Run4 ep8_bestval:
  - f4   : fused features (input to the DANN domain head, 512-d)
  - n4   : noise/Bayar branch stage-4 features (512-d)  [should be MORE domain-invariant]
  - e4   : edge/RGB branch stage-4 features (512-d)      [should be MORE domain-encoding]

We also measure the seg-head authentic false-positive rate per domain to link
the shortcut to the observed carpet-bombing (fires on RTM-domain regardless of tampering).

Read-only w.r.t. model/data. Writes results to reports/action0_shortcut_probe.json.
"""
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
N_PER_DOMAIN = 250
BATCH = 8
SEED = 42

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def load_model():
    model = MVSSNetLite().to(DEVICE)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def sample_pairs():
    manifest = json.load(open('reports/manifest.json'))
    by_dom = defaultdict(list)
    for s in manifest['test']:
        if s['dataset'] in DOMAINS:
            by_dom[s['dataset']].append((s['image'], s['mask']))
    pairs, dom_labels = [], []
    for d in DOMAINS:
        lst = by_dom[d]
        random.shuffle(lst)
        chosen = lst[:N_PER_DOMAIN]
        pairs.extend(chosen)
        dom_labels.extend([d] * len(chosen))
    return pairs, dom_labels


def gap(t):
    # global average pool over spatial dims -> (B, C); flatten(1) for safety
    return t.flatten(2).mean(dim=2)


def main():
    if not os.path.exists(CKPT):
        raise SystemExit(f"Checkpoint not found: {CKPT}")
    print(f"Device: {DEVICE}  |  checkpoint: {CKPT}")
    model = load_model()

    # hook the three feature sites
    feats = {}
    h1 = model.backbone.fuse4.register_forward_hook(lambda m, i, o: feats.__setitem__('f4', o.detach()))
    h2 = model.backbone.noise_layer4.register_forward_hook(lambda m, i, o: feats.__setitem__('n4', o.detach()))
    h3 = model.backbone.edge_layer4.register_forward_hook(lambda m, i, o: feats.__setitem__('e4', o.detach()))

    pairs, dom_labels = sample_pairs()
    print("Sampled per domain:", dict(Counter(dom_labels)))

    ds = ForgeryDataset(pairs, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=2)

    F4, N4, E4 = [], [], []
    # per-domain seg-head behaviour
    seg_fire, seg_forged_gt, seg_n = defaultdict(int), defaultdict(int), defaultdict(int)
    seg_fire_forged, seg_fire_auth = defaultdict(int), defaultdict(int)
    n_auth, n_forged = defaultdict(int), defaultdict(int)

    idx = 0
    with torch.no_grad():
        for imgs, masks, edges in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            seg, edge = model(imgs)  # lambda_p None -> (seg, edge); hooks fill feats
            F4.append(gap(feats['f4']).cpu().numpy())
            N4.append(gap(feats['n4']).cpu().numpy())
            E4.append(gap(feats['e4']).cpu().numpy())

            prob = torch.sigmoid(seg)
            fired = (prob.flatten(1) > 0.5).any(dim=1).cpu().numpy()
            forged_gt = (masks.flatten(1).sum(dim=1) > 0).cpu().numpy()
            bs = imgs.shape[0]
            for j in range(bs):
                d = dom_labels[idx + j]
                seg_n[d] += 1
                if forged_gt[j]:
                    n_forged[d] += 1
                    if fired[j]: seg_fire_forged[d] += 1
                else:
                    n_auth[d] += 1
                    if fired[j]: seg_fire_auth[d] += 1
            idx += bs

    h1.remove(); h2.remove(); h3.remove()

    F4 = np.concatenate(F4); N4 = np.concatenate(N4); E4 = np.concatenate(E4)
    y = np.array(dom_labels)
    print(f"Feature matrices: f4{F4.shape} n4{N4.shape} e4{E4.shape}")

    results = {'n_per_domain': N_PER_DOMAIN, 'domains': DOMAINS, 'chance': round(1/len(DOMAINS), 4)}

    # ---- linear domain probe on each feature site ----
    probe_out = {}
    for name, X in [('f4_fused', F4), ('n4_noise_branch', N4), ('e4_rgb_branch', E4)]:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = accuracy_score(yte, pred)
        cm = confusion_matrix(yte, pred, labels=DOMAINS)
        probe_out[name] = {'domain_acc': round(float(acc), 4),
                           'confusion': cm.tolist()}
        print(f"\n[{name}] linear domain-classification accuracy = {acc:.3f}  (chance {1/len(DOMAINS):.3f})")
        print("  confusion (rows=true, cols=pred; order=%s):" % DOMAINS)
        for r, row in zip(DOMAINS, cm):
            print(f"    {r:>10s}: {row.tolist()}")
    results['domain_probe'] = probe_out

    # ---- per-domain seg-head firing (carpet-bombing evidence) ----
    print("\nPer-domain seg-head firing @0.5 (fires anywhere in image):")
    seg_summary = {}
    for d in DOMAINS:
        auth_fp = seg_fire_auth[d] / max(1, n_auth[d])
        forged_rec = seg_fire_forged[d] / max(1, n_forged[d])
        seg_summary[d] = {
            'n': seg_n[d], 'n_auth': n_auth[d], 'n_forged': n_forged[d],
            'auth_fp_rate': round(auth_fp, 4), 'forged_recall': round(forged_rec, 4),
            'fire_auth': seg_fire_auth[d], 'fire_forged': seg_fire_forged[d],
        }
        print(f"  {d:>10s}: n={seg_n[d]:4d} auth={n_auth[d]:4d} forged={n_forged[d]:4d} "
              f"| authFP={auth_fp*100:5.1f}%  forgedRecall={forged_rec*100:5.1f}%")
    results['seg_head_firing'] = seg_summary

    os.makedirs('reports', exist_ok=True)
    json.dump(results, open('reports/action0_shortcut_probe.json', 'w'), indent=2)
    print("\nSaved reports/action0_shortcut_probe.json")

    # ---- verdict ----
    f4_acc = probe_out['f4_fused']['domain_acc']
    n4_acc = probe_out['n4_noise_branch']['domain_acc']
    e4_acc = probe_out['e4_rgb_branch']['domain_acc']
    print("\n===== VERDICT =====")
    print(f"f4 domain acc {f4_acc:.3f} | e4(RGB) {e4_acc:.3f} | n4(noise) {n4_acc:.3f} | chance {1/len(DOMAINS):.3f}")
    if f4_acc > 0.9:
        print("-> Shortcut CONFIRMED: fused features are near-perfectly domain-linear-separable.")
    if e4_acc - n4_acc > 0.05:
        print("-> RGB branch encodes domain MORE than noise branch -> supports 'use domain-invariant cue' fix.")
    elif n4_acc - e4_acc > 0.05:
        print("-> Noise branch encodes domain more than RGB (unexpected) -> revisit fix motivation.")
    else:
        print("-> RGB vs noise branch domain-encoding similar -> fix motivation needs a different angle.")


if __name__ == '__main__':
    main()
