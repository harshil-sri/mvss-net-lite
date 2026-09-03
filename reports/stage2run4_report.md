# Stage 2 Run 4 — Diagnostic Report

## 1. End Condition
**Early stopping at epoch 28.**
The training halted precisely at Epoch 28 because the validation loss did not improve beyond its minimum (`3.0783`) after a patience of 20 epochs (achieved at Epoch 8).

## 2. Per-Epoch Metrics
Full history table (including Seg/Edge losses, Learning Rate, Domain Accuracy, and Lambda values):

| Epoch | Train Seg | Train Edge | Train Total | Val Seg | Val Edge | Val Total | LR | Domain Acc | Lambda |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2.0969 | 2.0716 | 4.1857 | 1.6590 | 1.6388 | 3.2979 | 9.99e-05 | 94.07% | ~0.0624 |
| 2 | 1.8853 | 1.8976 | 3.7953 | 1.6120 | 1.5922 | 3.2043 | 9.98e-05 | 96.09% | ~0.1244 |
| 3 | 1.8572 | 1.8860 | 3.7589 | 1.5735 | 1.5787 | 3.1523 | 9.96e-05 | 94.63% | ~0.1856 |
| 4 | 1.8416 | 1.8865 | 3.7588 | 1.5519 | 1.5588 | 3.1108 | 9.93e-05 | 85.45% | ~0.2449 |
| 5 | 1.7976 | 1.8378 | 3.6663 | 1.5492 | 1.5615 | 3.1107 | 9.90e-05 | 85.82% | ~0.3019 |
| 6 | 1.8064 | 1.8384 | 3.6911 | 1.5731 | 1.5663 | 3.1394 | 9.86e-05 | 75.82% | ~0.3584 |
| 7 | 1.7912 | 1.8165 | 3.6644 | 1.6093 | 1.5721 | 3.1814 | 9.81e-05 | 68.20% | ~0.4116 |
| 8 | 1.7532 | 1.7928 | 3.6044 | 1.5309 | 1.5473 | 3.0782 | 9.75e-05 | 68.84% | ~0.4621 |
| 9 | 1.7521 | 1.7880 | 3.5997 | 1.5680 | 1.5581 | 3.1261 | 9.69e-05 | 68.04% | ~0.5098 |
| 10 | 1.7376 | 1.7756 | 3.5718 | 1.5488 | 1.5469 | 3.0958 | 9.62e-05 | 69.55% | ~0.5546 |
| 11 | 1.7242 | 1.7626 | 3.5438 | 1.5646 | 1.5543 | 3.1189 | 9.54e-05 | 71.76% | ~0.5966 |
| 12 | 1.6948 | 1.7406 | 3.4964 | 1.6468 | 1.5906 | 3.2375 | 9.46e-05 | 67.10% | ~0.6351 |
| 13 | 1.6692 | 1.7208 | 3.4522 | 1.6283 | 1.5743 | 3.2026 | 9.36e-05 | 67.27% | ~0.6710 |
| 14 | 1.6979 | 1.7349 | 3.4954 | 1.6350 | 1.6185 | 3.2536 | 9.27e-05 | 67.08% | ~0.7039 |
| 15 | 1.6464 | 1.7083 | 3.4176 | 1.7690 | 1.6061 | 3.3751 | 9.16e-05 | 64.40% | ~0.7341 |
| 16 | 1.6351 | 1.6911 | 3.3909 | 1.6140 | 1.5749 | 3.1889 | 9.05e-05 | 65.38% | ~0.7615 |
| 17 | 1.6274 | 1.6904 | 3.3796 | 1.6846 | 1.5960 | 3.2807 | 8.93e-05 | 66.54% | ~0.7866 |
| 18 | 1.6204 | 1.6866 | 3.3701 | 1.6097 | 1.5876 | 3.1974 | 8.81e-05 | 66.50% | ~0.8092 |
| 19 | 1.6043 | 1.6686 | 3.3347 | 1.6531 | 1.5841 | 3.2372 | 8.68e-05 | 66.51% | ~0.8298 |
| 20 | 1.5905 | 1.6602 | 3.3162 | 1.6803 | 1.6280 | 3.3084 | 8.55e-05 | 64.65% | ~0.8483 |
| 21 | 1.5664 | 1.6417 | 3.2732 | 1.7762 | 1.6741 | 3.4503 | 8.41e-05 | 64.72% | ~0.8649 |
| 22 | 1.5867 | 1.6494 | 3.3002 | 1.7607 | 1.6519 | 3.4126 | 8.26e-05 | 65.96% | ~0.8798 |
| 23 | 1.5594 | 1.6344 | 3.2603 | 1.6748 | 1.5970 | 3.2719 | 8.11e-05 | 63.93% | ~0.8931 |
| 24 | 1.5308 | 1.6044 | 3.1997 | 1.7356 | 1.6440 | 3.3796 | 7.95e-05 | 65.65% | ~0.9051 |
| 25 | 1.5225 | 1.5965 | 3.1856 | 1.7800 | 1.6673 | 3.4473 | 7.80e-05 | 63.60% | ~0.9158 |
| 26 | 1.5101 | 1.5880 | 3.1648 | 1.7037 | 1.6220 | 3.3257 | 7.63e-05 | 63.63% | ~0.9253 |
| 27 | 1.5057 | 1.5817 | 3.1520 | 1.7628 | 1.6713 | 3.4341 | 7.46e-05 | 64.92% | ~0.9338 |
| 28 | 1.4805 | 1.5635 | 3.1119 | 1.9325 | 1.7384 | 3.6709 | 7.29e-05 | 62.50% | ~0.9413 |

## 3. Checkpoints on Disk
The following checkpoint files currently exist from Stage 2 Run 4 (due to the rotation logic retaining every 5th epoch plus the overall best):
- `model/checkpoints/stage2run4_mvss_lite_ep15.pt`
- `model/checkpoints/stage2run4_mvss_lite_ep20.pt`
- `model/checkpoints/stage2run4_mvss_lite_ep25.pt`
- `model/checkpoints/stage2run4_mvss_lite_ep8_bestval.pt` (`_bestval` checkpoint)

*(Note: There were two initial dry-run checkpoints stored prior to the main run launching, but they are from the short preflight script: `stage2run4_dryrun_mvss_lite_ep1_bestval.pt` and `stage2run4_dryrun_mvss_lite_ep2.pt`).*

## 4. Domain-Split Specificity Probe
Using 200 untouched, authentic samples from the RTM and MIDV500 test sets across historical and current models:

| Model | RTM-Auth FP (0.5) | RTM-Auth FP (0.9) | MIDV500-Auth FP (0.5) | MIDV500-Auth FP (0.9) |
|---|---|---|---|---|
| **Run 4 `ep8_bestval`** | 159/200 (79.5%) | 127/200 (63.5%) | 24/200 (12.0%) | 1/200 (0.5%) |
| **Run 4 `ep25`** | 155/200 (77.5%) | 123/200 (61.5%) | 22/200 (11.0%) | 3/200 (1.5%) |
| **Run 3 `ep10_bestval`** | 164/200 (82.0%) | 126/200 (63.0%) | 32/200 (16.0%) | 4/200 (2.0%) |
| **Run 2 `ep40`** | 155/200 (77.5%) | 133/200 (66.5%) | 8/200 (4.0%) | 0/200 (0.0%) |

## 5. Convergence Points
- **Validation Loss Minimum**: Hit its absolute bottom at **Epoch 8** (`3.0783`).
- **Domain Accuracy Plateau**: Domain accuracy plummeted steeply from `96.09%` (Epoch 2) down to `68.20%` (Epoch 7), where it visibly plateaued and stabilized. From Epoch 7 out to Epoch 28, it hovered rigidly in the `62.5%` - `69.5%` band without ever completely collapsing to ~50% random chance level.

## 6. Monitor Log
```text
[2026-08-30 22:00:00] Initial check-in. Process is ALIVE (Scanning dataset). Epoch 1 hasn't started yet.
[2026-08-30 22:30:00] Epoch 1 completed. Process is ALIVE.
Train Total: 4.1858 | Val Total: 3.2979
LR: 0.000100 | Domain Acc: 94.07% | Lambda_p: ~0.0624
[2026-08-30 23:00:00] Epoch 2 completed. Process is ALIVE.
Train Total: 3.7954 | Val Total: 3.2043
LR: 0.000100 | Domain Acc: 96.09% | Lambda_p: ~0.1244
[2026-08-30 23:30:00] Epoch 4 completed. Process is ALIVE.
Train Total: 3.7588 | Val Total: 3.1108
LR: 0.000100 | Domain Acc: 85.45% | Lambda_p: ~0.2449
[2026-08-31 00:00:00] Epoch 6 completed. Process is ALIVE.
Train Total: 3.6911 | Val Total: 3.1395
LR: 0.000099 | Domain Acc: 75.82% | Lambda_p: ~0.3584
[2026-08-31 00:30:00] Epoch 7 completed. Process is ALIVE.
Train Total: 3.6644 | Val Total: 3.1815
LR: 0.000099 | Domain Acc: 68.20% | Lambda_p: ~0.4116
[2026-08-31 01:00:00] Epoch 9 completed. Process is ALIVE.
Train Total: 3.5998 | Val Total: 3.1262
LR: 0.000098 | Domain Acc: 68.04% | Lambda_p: ~0.5098
[2026-08-31 01:30:00] Epoch 10 completed. Process is ALIVE.
Train Total: 3.5719 | Val Total: 3.0959
LR: 0.000097 | Domain Acc: 69.55% | Lambda_p: ~0.5546
[2026-08-31 02:00:00] Epoch 12 completed. Process is ALIVE.
Train Total: 3.4965 | Val Total: 3.2375
LR: 0.000095 | Domain Acc: 67.10% | Lambda_p: ~0.6351
[2026-08-31 02:30:00] Epoch 14 completed. Process is ALIVE.
Train Total: 3.4954 | Val Total: 3.2536
LR: 0.000094 | Domain Acc: 67.08% | Lambda_p: ~0.7039
[2026-08-31 03:00:00] Epoch 15 completed. Process is ALIVE.
Train Total: 3.4177 | Val Total: 3.3752
LR: 0.000093 | Domain Acc: 64.40% | Lambda_p: ~0.7341
[2026-08-31 03:30:00] Epoch 17 completed. Process is ALIVE.
Train Total: 3.3796 | Val Total: 3.2807
LR: 0.000091 | Domain Acc: 66.54% | Lambda_p: ~0.7866
[2026-08-31 04:00:00] Epoch 19 completed. Process is ALIVE.
Train Total: 3.3347 | Val Total: 3.2373
LR: 0.000088 | Domain Acc: 66.51% | Lambda_p: ~0.8298
[2026-08-31 04:30:00] Epoch 20 completed. Process is ALIVE.
Train Total: 3.3163 | Val Total: 3.3084
LR: 0.000087 | Domain Acc: 64.65% | Lambda_p: ~0.8483
[2026-08-31 05:00:00] Epoch 22 completed. Process is ALIVE.
Train Total: 3.3002 | Val Total: 3.4127
LR: 0.000084 | Domain Acc: 65.96% | Lambda_p: ~0.8798
[2026-08-31 05:30:00] Epoch 24 completed. Process is ALIVE.
Train Total: 3.1998 | Val Total: 3.3797
LR: 0.000081 | Domain Acc: 65.65% | Lambda_p: ~0.9051
[2026-08-31 06:00:00] Epoch 25 completed. Process is ALIVE.
Train Total: 3.1856 | Val Total: 3.4474
LR: 0.000080 | Domain Acc: 63.60% | Lambda_p: ~0.9158
System: CPU ~25% | RAM 2032/5924MB | GPU Util ~1-30% | GPU Temp 53C | GPU Mem 3523MiB
[2026-08-31 06:30:00] Epoch 26 completed. Process is ALIVE.
Train Total: 3.1648 | Val Total: 3.3258
LR: 0.000078 | Domain Acc: 63.63% | Lambda_p: ~0.9253
System: CPU ~13% | RAM 2043/5924MB | GPU Util ~1-30% | GPU Temp 56C | GPU Mem 3622MiB
```

## 7. Exact Command Executed
```bash
tmux new-session -d -s stage2run4 'cd ~/mvss-net-lite && /home/harshil/.venvs/global/bin/python3 -u -m model.train \
  --datasets RTM MIDV500 \
  --epochs 80 \
  --stage-name stage2run4 \
  --init-weights model/checkpoints/stage1_mvss_lite_ep45.pt \
  --use-balanced-sampler \
  --lr-schedule cosine \
  --early-stopping-patience 20 \
  --domain-adversarial \
  2>&1 | tee reports/stage2run4_train.log'
```
