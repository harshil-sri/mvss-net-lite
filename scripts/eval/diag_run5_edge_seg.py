"""Separate edge-head vs seg-head behaviour on authentic MIDV500, Run4 vs Run5."""
import torch
import json
import numpy as np
from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

device = 'cuda'

m = json.load(open('reports/manifest.json'))
pairs = [(s['image'], s['mask']) for s in m['test'] if s['dataset'] == 'MIDV500'][:60]


def stats(path):
    model = MVSSNetLite().to(device)
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck['model_state_dict'], strict=False)
    model.eval()
    ds = ForgeryDataset(pairs, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    edge_max = []
    big_09 = 0
    seg_means = []
    edge_means = []
    with torch.no_grad():
        for imgs, masks, edges in loader:
            imgs = imgs.to(device)
            ps, pe, _ = model(imgs, lambda_p=0.0)
            p = torch.sigmoid(pe)
            psg = torch.sigmoid(ps)
            edge_max.extend(p.flatten(1).max(dim=1).values.cpu().tolist())
            big_09 += int((((p.flatten(1) > 0.9).sum(dim=1) > 50).sum().item()))
            seg_means.extend(psg.flatten(1).mean(dim=1).cpu().tolist())
            edge_means.extend(p.flatten(1).mean(dim=1).cpu().tolist())
    return {
        'edge_max_mean': np.mean(edge_max), 'edge_max_worst': max(edge_max),
        'imgs_gt50px_09': big_09, 'n': len(edge_max),
        'seg_mean': np.mean(seg_means), 'edge_mean': np.mean(edge_means),
    }


for path, label in [
    ('model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt', 'Run4'),
    ('model/checkpoints/stage2run5_mvss_lite_ep5.pt', 'Run5 ep5'),
    ('model/checkpoints/stage2run5_mvss_lite_ep11_bestval.pt', 'Run5 ep11'),
    ('model/checkpoints/stage2run5_mvss_lite_ep30.pt', 'Run5 ep30'),
]:
    s = stats(path)
    print(f"{label}: edge_max_mean={s['edge_max_mean']:.3f} worst={s['edge_max_worst']:.3f} "
          f">50px@0.9={s['imgs_gt50px_09']}/{s['n']} | seg_head_mean_sig={s['seg_mean']:.4f} "
          f"edge_head_mean_sig={s['edge_mean']:.4f}")