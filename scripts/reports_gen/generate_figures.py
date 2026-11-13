import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Set global style
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['axes.titlesize'] = 11
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['xtick.labelsize'] = 9
matplotlib.rcParams['ytick.labelsize'] = 9

# ============================================================================
# Figure 1: Base-rate causal sweep
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

# Data
b = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
mvss_auth_fp = np.array([0.000, 0.586, 0.589, 0.670, 0.748])
resnet_auth_fp = np.array([0.168, 0.796, 0.731, 0.977, 1.000])

# Plot lines
ax.plot(b, mvss_auth_fp, 'o-', label='MVSS-Net-Lite (ρ=1.0)', color='#1f77b4', linewidth=2, markersize=6)
ax.plot(b, resnet_auth_fp, 's--', label='ResNet-UNet/FPN (ρ=0.90)', color='#ff7f0e', linewidth=2, markersize=6)

# Reference line (y = x)
ax.plot([0, 1], [0, 1], 'k:', linewidth=1.5, alpha=0.5, label='y = x (perfect tracking)')

# Labels and title
ax.set_xlabel('Training Forged Base-Rate')
ax.set_ylabel('Authentic FP on Held-Out Eval')
ax.set_title('Authentic FP Tracks Training Base-Rate')

# Grid and legend
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper left')

# Tight layout
plt.tight_layout()
fig.savefig('/home/harshil/mvss-net-lite/workspace/figures/fig_base_rate_causal_sweep.png', dpi=150)
plt.close(fig)

# ============================================================================
# Figure 2: Domain separability at initialization
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

# Data
architectures = ['MVSS-Net-Lite', 'ResNet-UNet/FPN']
random_init = [0.80, 0.791]
imagenet_init = [np.nan, 0.922]  # MVSS-Net-Lite has no ImageNet init reported

# Bar positions
x = np.arange(len(architectures))
width = 0.35

# Plot bars
ax.bar(x - width/2, random_init, width, label='Random Init (f4)', color='#1f77b4', edgecolor='black', linewidth=0.5)
ax.bar(x + width/2, imagenet_init, width, label='ImageNet Init (f4)', color='#ff7f0e', edgecolor='black', linewidth=0.5)

# Chance line
ax.axhline(y=0.25, color='gray', linestyle='--', linewidth=1.5, label='Chance (4 domains)')

# Labels and title
ax.set_xlabel('Architecture')
ax.set_ylabel('Domain Separability (f4)')
ax.set_title('Domain Separability at Random Initialization')
ax.set_xticks(x)
ax.set_xticklabels(architectures)

# Grid and legend
ax.grid(True, linestyle='--', alpha=0.6, axis='y')
ax.legend(loc='lower right')

# Tight layout
plt.tight_layout()
fig.savefig('/home/harshil/mvss-net-lite/workspace/figures/fig_domain_separability_init.png', dpi=150)
plt.close(fig)

# ============================================================================
# Figure 3: DG methods degradation
# ============================================================================
fig, ax = plt.subplots(figsize=(6, 4.5))

# Data
methods = ['ERM', 'DANN', 'Group-DRO', 'DetectionHead']
auth_fp = [0.34, 0.36, 1.00, 0.29]

# Colors: Group-DRO should stand out as catastrophic (red)
colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c']

# Plot bars
ax.bar(methods, auth_fp, color=colors, edgecolor='black', linewidth=0.5)

# Labels and title
ax.set_xlabel('Method')
ax.set_ylabel('Auth-FP@85% (Lower is Better)')
ax.set_title('Performance Degradation of DG Methods (CASIA)')

# Grid and legend
ax.grid(True, linestyle='--', alpha=0.6, axis='y')

# Tight layout
plt.tight_layout()
fig.savefig('/home/harshil/mvss-net-lite/workspace/figures/fig_dg_methods_degradation.png', dpi=150)
plt.close(fig)

print("All figures generated successfully.")
