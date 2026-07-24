import torch
import numpy as np
import os

torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)

from model.network import MVSSNetLite
from run_domain_split_pilot import extract_split_probe_images

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    rtm_tensor, _ = extract_split_probe_images()
    model = MVSSNetLite().to(device)
    model.eval()
    
    max_probs = []
    with torch.no_grad():
        for i in range(rtm_tensor.size(0)):
            img = rtm_tensor[i].unsqueeze(0)
            _, pred_edge = model(img)
            prob = torch.sigmoid(pred_edge).squeeze().cpu().numpy()
            max_probs.append(prob.max())
            
    peak_val = np.max(max_probs)
    peak_idx = np.argmax(max_probs)
    
    print(f"\n--- UNTRAINED PROBE (RTM) ---")
    print(f"Avg Max Prob: {np.mean(max_probs):.4f}")
    print(f"PEAK PROB: {peak_val:.4f} on Image #{peak_idx}")
    print(f"Top 3 Peak Images: {np.argsort(max_probs)[-3:][::-1]}")
