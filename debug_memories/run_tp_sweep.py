import torch
import numpy as np
import os

from model.network import MVSSNetLite
from data_pipeline.dataset_loader import get_dataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_forged_probes():
    print("Extracting domain-split probe images (RTM FORGED)...")
    _, rtm_val, _ = get_dataloader(['RTM'], batch_size=1, is_train=False, return_splits=True)
    
    rtm_probes = []
    rtm_edges = []
    
    for imgs, masks, edges in rtm_val:
        if masks[0].sum() > 0:
            rtm_probes.append(imgs[0])
            rtm_edges.append(edges[0])
            if len(rtm_probes) == 20:
                break
                
    print(f"Extraction complete. Found {len(rtm_probes)} RTM FORGED images.")
    return torch.stack(rtm_probes).to(device) if rtm_probes else None, torch.stack(rtm_edges).to(device) if rtm_edges else None

def run_tp_probe(model, img_tensor, edge_tensor, step_name):
    if img_tensor is None:
        return
        
    model.eval()
    
    # Track metrics across thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    tp_pixels = {t: [] for t in thresholds}
    
    with torch.no_grad():
        for i in range(img_tensor.size(0)):
            img = img_tensor[i].unsqueeze(0)
            gt_edge = edge_tensor[i].unsqueeze(0)
            
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            gt = gt_edge.squeeze().cpu().numpy()
            
            for t in thresholds:
                pred_binary = (prob > t)
                # True positive pixels: predicted as edge AND is actually edge
                tp = np.logical_and(pred_binary, gt > 0.5).sum()
                tp_pixels[t].append(tp)
    
    print(f"\n--- TP PROBE AT {step_name} (RTM FORGED) ---")
    for t in thresholds:
        avg_tp = np.mean(tp_pixels[t])
        print(f"Threshold {t}: Avg TP Pixels: {avg_tp:.2f}")
    print("---------------------------\n")

def main():
    img_tensor, edge_tensor = extract_forged_probes()
    
    for epoch in [3, 5]:
        chkpt_path = f"model/checkpoints/stage2_mvss_lite_ep{epoch}.pt"
        if not os.path.exists(chkpt_path):
            print(f"Checkpoint not found: {chkpt_path}")
            continue
            
        model = MVSSNetLite().to(device)
        model.load_state_dict(torch.load(chkpt_path, map_location=device))
        run_tp_probe(model, img_tensor, edge_tensor, f"Epoch {epoch}")

if __name__ == '__main__':
    main()
