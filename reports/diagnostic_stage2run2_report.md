# Stage 2 Run 2 — Diagnostic Report Only

## 1. Checkpoints actually on disk from this run
The following checkpoints are currently on disk (saved in `model/checkpoints/`):
- `stage2run2_mvss_lite_ep5.pt` (Epoch 5, 2026-08-28 00:34:21 UTC)
- `stage2run2_mvss_lite_ep40.pt` (Epoch 40, 2026-08-28 10:19:13 UTC)
- `stage2run2_mvss_lite_ep45.pt` (Epoch 45, 2026-08-28 11:43:09 UTC)
- `stage2run2_mvss_lite_ep50.pt` (Epoch 50, 2026-08-28 13:06:54 UTC)

**Rotation / Guard Logic Outcome:**
- Checkpoints were saved every 5 epochs.
- The rotation logic deleted epochs 10, 15, 20, 25, 30, and 35.
- The window limit strictly kept the last 3 checkpoints (`ep40`, `ep45`, `ep50`).
- Epoch 5 was explicitly **spared-as-best** because it had the lowest total validation loss of any *saved* checkpoint (3.1414).

**Explicit Note on True Bottom:** 
The absolute true val-loss bottom occurred at **Epoch 4** (3.1286). Because the script only saved checkpoints on multiples of 5, Epoch 4 was **not** among the saved checkpoints.

---

## 2. Val loss by epoch, exact numbers

| Epoch | Train Total | Train Seg | Train Edge | Val Total | Val Seg | Val Edge |
|---|---|---|---|---|---|---|
| 1 | 4.2186 | 2.1327 | 2.0859 | 3.2187 | 1.6085 | 1.6102 |
| 2 | 3.8218 | 1.9133 | 1.9084 | 3.1459 | 1.5702 | 1.5756 |
| 3 | 3.7726 | 1.8826 | 1.8901 | 3.1537 | 1.5768 | 1.5769 |
| 4 | 3.6734 | 1.8273 | 1.8461 | 3.1286 | 1.5650 | 1.5636 |
| 5 | 3.6127 | 1.7949 | 1.8177 | 3.1414 | 1.5804 | 1.5610 |
| 6 | 3.5685 | 1.7735 | 1.7951 | 3.1294 | 1.5673 | 1.5621 |
| 7 | 3.5394 | 1.7548 | 1.7847 | 3.1339 | 1.5702 | 1.5637 |
| 8 | 3.5219 | 1.7428 | 1.7790 | 3.2471 | 1.6532 | 1.5939 |
| 9 | 3.5201 | 1.7425 | 1.7776 | 3.2250 | 1.6400 | 1.5850 |
| 10 | 3.4456 | 1.7021 | 1.7435 | 3.2660 | 1.6600 | 1.6060 |
| 11 | 3.3815 | 1.6688 | 1.7127 | 3.1568 | 1.5928 | 1.5639 |
| 12 | 3.3686 | 1.6602 | 1.7085 | 3.2733 | 1.6712 | 1.6021 |
| 13 | 3.3590 | 1.6555 | 1.7035 | 3.4674 | 1.8033 | 1.6641 |
| 14 | 3.3405 | 1.6425 | 1.6979 | 3.3093 | 1.7070 | 1.6023 |
| 15 | 3.3686 | 1.6624 | 1.7061 | 3.2486 | 1.6475 | 1.6010 |
| 16 | 3.2782 | 1.6160 | 1.6621 | 3.3833 | 1.7547 | 1.6286 |
| 17 | 3.2541 | 1.5984 | 1.6556 | 3.5640 | 1.8761 | 1.6879 |
| 18 | 3.2494 | 1.5949 | 1.6545 | 3.3410 | 1.7210 | 1.6199 |
| 19 | 3.2233 | 1.5822 | 1.6410 | 3.5222 | 1.8399 | 1.6823 |
| 20 | 3.2055 | 1.5694 | 1.6361 | 3.6226 | 1.9030 | 1.7195 |
| 21 | 3.2262 | 1.5821 | 1.6441 | 3.6477 | 1.9200 | 1.7276 |
| 22 | 3.1575 | 1.5428 | 1.6146 | 3.5527 | 1.8702 | 1.6825 |
| 23 | 3.1416 | 1.5392 | 1.6024 | 3.5992 | 1.8831 | 1.7162 |
| 24 | 3.1349 | 1.5352 | 1.5998 | 3.5234 | 1.8372 | 1.6861 |
| 25 | 3.0875 | 1.5045 | 1.5830 | 3.4584 | 1.7912 | 1.6672 |
| 26 | 3.0785 | 1.5008 | 1.5777 | 3.5746 | 1.8641 | 1.7106 |
| 27 | 3.0920 | 1.5097 | 1.5823 | 3.5328 | 1.8209 | 1.7119 |
| 28 | 3.0663 | 1.4950 | 1.5714 | 3.7862 | 2.0078 | 1.7784 |
| 29 | 3.0302 | 1.4757 | 1.5546 | 3.8575 | 2.0601 | 1.7974 |
| 30 | 3.0381 | 1.4818 | 1.5563 | 3.7433 | 1.9759 | 1.7673 |
| 31 | 3.0068 | 1.4650 | 1.5418 | 3.8220 | 2.0312 | 1.7908 |
| 32 | 3.0238 | 1.4734 | 1.5504 | 3.7228 | 1.9845 | 1.7383 |
| 33 | 2.9972 | 1.4571 | 1.5400 | 4.0729 | 2.1972 | 1.8757 |
| 34 | 2.9995 | 1.4602 | 1.5392 | 3.9590 | 2.1178 | 1.8412 |
| 35 | 2.9582 | 1.4380 | 1.5202 | 4.0459 | 2.1692 | 1.8766 |
| 36 | 2.9467 | 1.4308 | 1.5160 | 4.3931 | 2.3790 | 2.0140 |
| 37 | 2.9685 | 1.4437 | 1.5248 | 4.2011 | 2.2629 | 1.9382 |
| 38 | 2.9469 | 1.4306 | 1.5163 | 4.2695 | 2.3332 | 1.9363 |
| 39 | 2.9242 | 1.4204 | 1.5038 | 4.0060 | 2.1506 | 1.8554 |
| 40 | 2.9197 | 1.4197 | 1.5000 | 4.2775 | 2.3349 | 1.9426 |
| 41 | 2.8782 | 1.3943 | 1.4839 | 4.4065 | 2.4146 | 1.9918 |
| 42 | 2.8781 | 1.3970 | 1.4811 | 4.2715 | 2.3268 | 1.9446 |
| 43 | 2.8563 | 1.3850 | 1.4714 | 4.2296 | 2.2870 | 1.9426 |
| 44 | 2.8734 | 1.3987 | 1.4747 | 4.3913 | 2.3668 | 2.0245 |
| 45 | 2.8539 | 1.3794 | 1.4746 | 4.6169 | 2.5446 | 2.0724 |
| 46 | 2.8607 | 1.3855 | 1.4752 | 4.0808 | 2.2109 | 1.8698 |
| 47 | 2.8686 | 1.3914 | 1.4772 | 4.1176 | 2.2046 | 1.9130 |
| 48 | 2.8219 | 1.3662 | 1.4557 | 4.8020 | 2.6457 | 2.1563 |
| 49 | 2.8110 | 1.3583 | 1.4527 | 4.6618 | 2.5345 | 2.1273 |
| 50 | 2.8163 | 1.3665 | 1.4497 | 4.3415 | 2.3759 | 1.9656 |

---

## 3. Domain-split specificity probe, at each saved checkpoint
*(Sample: 200 RTM authentic, 200 MIDV500 authentic)*

| Checkpoint | RTM Auth FP (Thresh 0.5) | RTM Auth FP (Thresh 0.9) | MIDV500 Auth FP (Thresh 0.5) | MIDV500 Auth FP (Thresh 0.9) |
|---|---|---|---|---|
| Epoch 5 | 184 / 200 | 133 / 200 | 47 / 200 | 1 / 200 |
| Epoch 40 | 155 / 200 | 133 / 200 | 8 / 200 | 0 / 200 |
| Epoch 45 | 185 / 200 | 152 / 200 | 58 / 200 | 8 / 200 |
| Epoch 50 | 153 / 200 | 135 / 200 | 10 / 200 | 4 / 200 |

---

## 4. Freeze configuration actually used
**Modules Frozen:**
- `noise_branch[stem+L1+L2+bayar]`
- `edge_branch[stem+L1+L2]`

**Modules Trainable:**
- `backbone[layer3+layer4 both branches]`
- `cbam_fusion[fuse1-4]`
- `decoder+heads`

**Parameter Counts:**
- TOTAL::trainable: 43,021,586
- TOTAL::frozen: 2,696,033
- TOTAL::all: 45,717,619
- *The fraction (2,696,033 / 45,717,619) exactly matches the run logs.*

**BatchNorm Status:**
There were 32 Frozen BatchNorm layers in the frozen portion. The training script explicitly iterated over them to call `keep_frozen_bn_eval(frozen_bn_layers)` *after* `model.train()` at the start of every epoch. Therefore, they were successfully kept in eval mode for the full 50-epoch run.

---

## 5. Sampler composition log, full run
Extracted from `reports/stage2run2_sampler_composition.log`. 

**Spot Checks (Batch 1 of specified epochs):**
- **Epoch 1:** RTM Forged: 25.00%, RTM Auth: 62.50%, MIDV500: 12.50%
- **Epoch 10:** RTM Forged: 37.50%, RTM Auth: 50.00%, MIDV500: 12.50%
- **Epoch 25:** RTM Forged: 50.00%, RTM Auth: 12.50%, MIDV500: 37.50%
- **Epoch 50:** RTM Forged: 37.50%, RTM Auth: 37.50%, MIDV500: 25.00%

**Global Average (summed across all samples in the entire 50-epoch run):**
- RTM Forged: 32.85%
- RTM Authentic: 33.67%
- MIDV500: 33.47%

**Flag:** The composition did **not** drift. Despite standard stochastic variance at the batch level (N=8), the target mass of ~1/3 each was successfully held throughout the run.

---

## 6. pos_weight values actually used
Calculated and logged at run initialization during dataset scan:
- **Seg pos_weight:** 233.67
- **Edge pos_weight:** 951.54

---

## 7. Any warnings, guard triggers, or anomalies in the run logs
- **Anomalies / Exceptions:** A comprehensive search of the logs for `warning`, `error`, `exception`, `nan`, `inf`, `restart`, `resume`, `guard`, or `halt` returned absolutely no results.
- **Disk Guard:** The free space check consistently logged `(free space after save: ~741.70GB)`. The threshold was never triggered.
- **Restarts:** None. The run executed from Epoch 1 to 50 in a single continuous session without interruption.

---

## 8. Exact training configuration as executed
**Executed Command:**
```bash
/home/harshil/.venvs/global/bin/python3 -u -m model.train \
  --datasets RTM MIDV500 \
  --epochs 50 \
  --stage-name stage2run2 \
  --init-weights model/checkpoints/stage1_mvss_lite_ep45.pt \
  --use-balanced-sampler \
  --freeze-early-layers
```

**Training Details Resolved:**
- **Optimizer:** `Adam` (Instantiated over the 43,021,586 trainable parameters only).
- **Learning Rate:** `0.000100` (Remained flat. No learning rate scheduler stepped the value down at any point during the run).
- **Batch Size:** 8
- **Total Steps:** 119,800 steps (2,396 batches/epoch × 50 epochs).
