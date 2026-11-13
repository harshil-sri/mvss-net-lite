#!/usr/bin/env python3
"""
Verify that generated figures are non-flat by checking RGB extrema.
"""
from PIL import Image
import numpy as np
import os

fig_dir = "/home/harshil/mvss-net-lite/workspace/figures"
fig_files = [
    "fig_base_rate_causal_sweep.png",
    "fig_domain_separability_init.png",
    "fig_decision_layer_pathology.png",
    "fig_dg_methods_degradation.png"
]

for fig_file in fig_files:
    fig_path = os.path.join(fig_dir, fig_file)
    img = Image.open(fig_path)
    img_array = np.array(img)
    
    # Calculate min/max RGB values
    if img_array.ndim == 3:  # RGB or RGBA
        r_min, r_max = img_array[:, :, 0].min(), img_array[:, :, 0].max()
        g_min, g_max = img_array[:, :, 1].min(), img_array[:, :, 1].max()
        b_min, b_max = img_array[:, :, 2].min(), img_array[:, :, 2].max()
        print(f"{fig_file}:")
        print(f"  R: min={r_min}, max={r_max}")
        print(f"  G: min={g_min}, max={g_max}")
        print(f"  B: min={b_min}, max={b_max}")
    elif img_array.ndim == 2:  # Grayscale
        gray_min, gray_max = img_array.min(), img_array.max()
        print(f"{fig_file}:")
        print(f"  Grayscale: min={gray_min}, max={gray_max}")
    else:
        print(f"{fig_file}: Unknown format")
    print()
