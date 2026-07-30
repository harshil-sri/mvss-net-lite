import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Create a dummy image of 1000x1000
mask = np.zeros((1000, 1000), dtype=np.uint8)

# Draw a thin 1-pixel horizontal line and a 1-pixel diagonal line
mask[500, 100:900] = 255
for i in range(100, 900):
    mask[i, i] = 255
    
mask_pil = Image.fromarray(mask)
crop_size = 256

# 1. NEAREST (Old Bug)
mask_nearest = mask_pil.resize((crop_size, crop_size), Image.NEAREST)
mask_nearest_np = (np.array(mask_nearest) > 127).astype(np.uint8)
nearest_pixels = mask_nearest_np.sum()

# 2. BILINEAR + >0 (The Fattening Bug)
mask_bilinear = mask_pil.resize((crop_size, crop_size), Image.BILINEAR)
mask_bilinear_np = (np.array(mask_bilinear) > 0).astype(np.uint8)
bilinear_pixels = mask_bilinear_np.sum()

# 3. Adaptive Max Pool 2D (True Max Pooling)
mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0) # (1, 1, 1000, 1000)
mask_max_t = F.adaptive_max_pool2d(mask_t, (crop_size, crop_size))
mask_max_np = (mask_max_t.squeeze().numpy() > 0).astype(np.uint8)
max_pixels = mask_max_np.sum()

print("Original 1000x1000 Mask Pixels:", (mask > 0).sum())
print("1. NEAREST downsample (old):", nearest_pixels)
print("2. BILINEAR >0 downsample (fattened):", bilinear_pixels)
print("3. Adaptive Max Pool (proper):", max_pixels)

# Check width of the horizontal line in the output
print("\nHorizontal Line Width Analysis at Col 128:")
print("NEAREST column sum:", mask_nearest_np[:, 128].sum())
print("BILINEAR column sum:", mask_bilinear_np[:, 128].sum())
print("MAX POOL column sum:", mask_max_np[:, 128].sum())
