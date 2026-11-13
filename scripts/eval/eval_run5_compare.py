"""Apples-to-apples authentic-FP comparison across runs on identical test splits."""
import torch
import json
import cv2
from model.network import MVSSNetLite
from data_pipeline.dataset_loader import ForgeryDataset

device = 'cuda'

def is_forged(s):
    mask = cv2.imread(s['mask'], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False
    return mask.sum() > 0

def eval_set(model, pairs):
    ds = ForgeryDataset(pairs, is_train=False)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2)
    fp05 = fp09 = n = 0
    with torch.no_grad():
        for imgs, masks, edges in loader:
            imgs = imgs.to(device)
            _, pe, _ = model(imgs, lambda_p=0.0)
            prob = torch.sigmoid(pe)
            fp05 += (prob > 0.5).any(dim=(1, 2, 3)).sum().item()
            fp09 += (prob > 0.9).any(dim=(1, 2, 3)).sum().item()
            n += len(imgs)
    return fp05, fp09, n

def load(path):
    m = MVSSNetLite().to(device)
    ck = torch.load(path, map_location=device)
    m.load_state_dict(ck['model_state_dict'], strict=False)
    m.eval()
    return m

def main():
    manifest = json.load(open('reports/manifest.json'))
    rtm_auth = [(s['image'], s['mask']) for s in manifest['test'] if s['dataset'] == 'RTM' and not is_forged(s)]
    rtm_forged = [(s['image'], s['mask']) for s in manifest['test'] if s['dataset'] == 'RTM' and is_forged(s)]
    midv_auth = [(s['image'], s['mask']) for s in manifest['test'] if s['dataset'] == 'MIDV500']
    midv_synth = [(s['image'], s['mask']) for s in manifest['test'] if s['dataset'] == 'MIDV500_synthetic' and is_forged(s)]

    checkpoints = [
        ("model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt", "Run4 ep8_bestval"),
        ("model/checkpoints/stage2run4_mvss_lite_ep25.pt", "Run4 ep25"),
        ("model/checkpoints/stage2run5_mvss_lite_ep5.pt", "Run5 ep5"),
        ("model/checkpoints/stage2run5_mvss_lite_ep11_bestval.pt", "Run5 ep11_bestval"),
        ("model/checkpoints/stage2run5_mvss_lite_ep30.pt", "Run5 ep30"),
    ]
    print(f"{'model':>22} | RTM-auth FP05   FP09 | MIDV-auth FP05  FP09 | RTM-forged rec05 | MIDVsynth rec05")
    print("-" * 108)
    for path, label in checkpoints:
        m = load(path)
        ra = eval_set(m, rtm_auth)
        ma = eval_set(m, midv_auth)
        rf = eval_set(m, rtm_forged)
        ms = eval_set(m, midv_synth)
        print(f"{label:>22} | {ra[0]}/{ra[2]} ({ra[0]/ra[2]*100:4.1f}%) {ra[1]}/{ra[2]} ({ra[1]/ra[2]*100:4.1f}%) | "
              f"{ma[0]}/{ma[2]} ({ma[0]/ma[2]*100:4.1f}%) {ma[1]}/{ma[2]} ({ma[1]/ma[2]*100:4.1f}%) | "
              f"{rf[0]/rf[2]*100:5.1f}% | {ms[0]/ms[2]*100:5.1f}%")

if __name__ == '__main__':
    main()