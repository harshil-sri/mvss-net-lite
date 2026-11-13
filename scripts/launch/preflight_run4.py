import math
import torch
import torch.nn as nn

# 1. Cosine LR check
model = nn.Linear(10, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
total_epochs = 80
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

print("--- 1. Cosine LR Check ---")
for epoch in range(total_epochs):
    if epoch in [0, 20, 40, 60, 79]:
        print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6e}")
    scheduler.step()

# 2. Lambda schedule check
print("\n--- 2. Lambda Schedule Check ---")
total_steps = 2396 * 80  # approx batches per epoch * 80
for p_frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
    global_step = int(p_frac * total_steps)
    p = global_step / float(total_steps)
    lambda_p = (2.0 / (1.0 + math.exp(-10.0 * p))) - 1.0
    print(f"p = {p:.2f} -> lambda_p = {lambda_p:.4f}")

# 3. Early-stopping logic check
print("\n--- 3. Early Stopping Logic Check ---")
es_patience = 20
es_counter = 0
global_best_val_loss = float('inf')

# Synthetic val loss sequence: minimum at epoch 5, then plateaus/rises
for epoch in range(1, 81):
    if epoch <= 5:
        val_loss = 5.0 - epoch  # 4.0, 3.0, 2.0, 1.0, 0.0
    else:
        val_loss = 0.5 + 0.1 * epoch  # Rises after epoch 5
        
    if val_loss < global_best_val_loss:
        global_best_val_loss = val_loss
        es_counter = 0
    else:
        es_counter += 1
        
    if es_counter >= es_patience:
        print(f"Early stopping triggered at epoch {epoch} (patience={es_patience} reached).")
        break
    if epoch == 80:
        print("Hit 80 epoch ceiling without early stopping.")

