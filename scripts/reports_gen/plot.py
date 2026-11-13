import matplotlib.pyplot as plt
import numpy as np

# Parse the lines
with open('reports/tmux_stage2run2.log', 'r') as f:
    lines = f.readlines()

train_seg, train_edge, train_tot = [], [], []
val_seg, val_edge, val_tot = [], [], []

for line in lines:
    if "TRAIN -> Avg Seg:" in line:
        parts = line.split('|')
        train_seg.append(float(parts[0].split(':')[1].strip()))
        train_edge.append(float(parts[1].split(':')[1].strip()))
        train_tot.append(float(parts[2].split(':')[1].strip()))
    elif "VAL   -> Avg Seg:" in line:
        parts = line.split('|')
        val_seg.append(float(parts[0].split(':')[1].strip()))
        val_edge.append(float(parts[1].split(':')[1].strip()))
        val_tot.append(float(parts[2].split(':')[1].strip()))

epochs = range(1, len(train_seg) + 1)

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(epochs, train_tot, label='Train Total', color='#1f77b4', linewidth=2, marker='o', markersize=4)
plt.plot(epochs, val_tot, label='Val Total', color='#d62728', linewidth=2, marker='o', markersize=4)
plt.title('Total Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(epochs, train_seg, label='Train Seg', color='#2ca02c', linewidth=2, marker='s', markersize=4)
plt.plot(epochs, val_seg, label='Val Seg', color='#ff7f0e', linewidth=2, marker='s', markersize=4)
plt.title('Segmentation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Seg Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(epochs, train_edge, label='Train Edge', color='#9467bd', linewidth=2, marker='^', markersize=4)
plt.plot(epochs, val_edge, label='Val Edge', color='#8c564b', linewidth=2, marker='^', markersize=4)
plt.title('Edge Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Edge Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('/home/harshil/.gemini/antigravity-cli/brain/c5e2e37d-2ff8-4c4a-b29e-552f13f1c0d6/detailed_loss_curve.png', dpi=300)
print("Plot saved.")
