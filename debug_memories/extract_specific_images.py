import torch
import cv2
import numpy as np

torch.manual_seed(42)
import random
random.seed(42)
np.random.seed(42)

from run_domain_split_pilot import extract_split_probe_images

if __name__ == '__main__':
    rtm_tensor, _ = extract_split_probe_images()
    
    for idx in [14, 19]:
        img = rtm_tensor[idx]
        img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"offender_image_{idx}.png", img_np)
        
    print("Saved offender_image_14.png and offender_image_19.png")
