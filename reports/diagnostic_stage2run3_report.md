# Stage 2 Run 3 — Diagnostic Report

## 1. Checkpoints actually on disk from this run
The following checkpoints are currently on disk (saved in `model/checkpoints/`):
- `stage2run3_mvss_lite_ep5.pt`
- `stage2run3_mvss_lite_ep10.pt`
- `stage2run3_mvss_lite_ep15.pt`
- `stage2run3_mvss_lite_ep10_bestval.pt`

**Rotation / Guard Logic Outcome:**
- Regular checkpoints were saved every 5 epochs.
- The rotation logic was configured to keep only the last 3 standard checkpoints, which is why 5, 10, and 15 are present (Epoch 18 halted before 20).
- The `_bestval.pt` checkpoint mechanism successfully preserved Epoch 10, which had the absolute lowest total validation loss of the run (3.1082).

---

## 2. Val loss by epoch, exact numbers

| Epoch | Train Total | Train Seg | Train Edge | Val Total | Val Seg | Val Edge |
|---|---|---|---|---|---|---|
| 1 | 4.1915 | 2.1108 | 2.0633 | 3.1690 | 1.5838 | 1.5851 |
| 2 | 3.8608 | 1.9137 | 1.9259 | 3.1629 | 1.5852 | 1.5776 |
| 3 | 3.7828 | 1.8645 | 1.8735 | 3.2093 | 1.6236 | 1.5857 |
| 4 | 3.7861 | 1.8598 | 1.8759 | 3.1837 | 1.6005 | 1.5833 |
| 5 | 3.7269 | 1.8270 | 1.8423 | 3.1729 | 1.5974 | 1.5755 |
| 6 | 3.6632 | 1.7907 | 1.8119 | 3.3351 | 1.6907 | 1.6444 |
| 7 | 3.6860 | 1.8012 | 1.8244 | 3.2388 | 1.6568 | 1.5820 |
| 8 | 3.6035 | 1.7519 | 1.7905 | 3.1136 | 1.5570 | 1.5567 |
| 9 | 3.5922 | 1.7509 | 1.7777 | 3.1852 | 1.6104 | 1.5748 |
| 10 | 3.5573 | 1.7346 | 1.7594 | 3.1082 | 1.5614 | 1.5467 |
| 11 | 3.5179 | 1.7099 | 1.7447 | 3.1555 | 1.5920 | 1.5634 |
| 12 | 3.4901 | 1.6942 | 1.7331 | 3.1553 | 1.5897 | 1.5657 |
| 13 | 3.5047 | 1.6935 | 1.7453 | 3.2182 | 1.6270 | 1.5913 |
| 14 | 3.4609 | 1.6778 | 1.7198 | 3.1705 | 1.6101 | 1.5604 |
| 15 | 3.4082 | 1.6432 | 1.7012 | 3.1702 | 1.6098 | 1.5604 |
| 16 | 3.3913 | 1.6346 | 1.6922 | 3.2416 | 1.6599 | 1.5817 |
| 17 | 3.3695 | 1.6258 | 1.6790 | 3.2962 | 1.6998 | 1.5964 |
| 18 | 3.3563 | 1.6160 | 1.6757 | 3.2125 | 1.6332 | 1.5792 |

---

## 3. Domain-split specificity probe, at each saved checkpoint
| Checkpoint | RTM Auth FP (Thresh 0.5) | RTM Auth FP (Thresh 0.9) | MIDV500 Auth FP (Thresh 0.5) | MIDV500 Auth FP (Thresh 0.9) |
|---|---|---|---|---|
| Epoch 5 | 167/200 | 109/200 | 13/200 | 1/200 |
| Epoch 10 | 164/200 | 126/200 | 32/200 | 4/200 |
| Epoch 15 | 167/200 | 137/200 | 28/200 | 4/200 |
| Epoch 10 (Best) | 164/200 | 126/200 | 32/200 | 4/200 |


---

## 4. Freeze configuration actually used
**Modules Frozen:**
- *None*. As specified in the run plan, early layer freezing was entirely disabled for this run to prevent confounding the interpretation of the Domain Adversarial loss.
- Total parameters trainable: 45,717,619 (100%)

---

## 5. Sampler composition log, full run
Extracted from `reports/stage2run3_sampler_composition.log`. 

**Spot Checks (Batch 1 of specified epochs):**
- **Epoch 1:** RTM Forged: 50.00%, RTM Auth: 12.50%, MIDV500: 37.50%
- **Epoch 10:** RTM Forged: 37.50%, RTM Auth: 25.00%, MIDV500: 37.50%
- **Epoch 18:** RTM Forged: 50.00%, RTM Auth: 12.50%, MIDV500: 37.50%

*(The composition remained properly balanced across classes using WeightedRandomSampler)*

---

## 6. pos_weight values actually used
Calculated and logged at run initialization during dataset scan:
- **Seg pos_weight:** 225.81
- **Edge pos_weight:** 935.30

---

## 7. Any warnings, guard triggers, or anomalies in the run logs
- **Anomalies / Exceptions:** No anomalous behaviors or NaNs occurred.
- **Early Stopping Halt:** The training was successfully halted at Epoch 18 (Patience = 8 was exhausted because the validation loss did not improve beyond the Epoch 10 minimum of 3.1082).
- **Domain Accuracy Drop:** Domain accuracy cleanly plummeted from 93.5% down to 66% as the GRL lambda increased, confirming the adversarial mechanism worked.

---

## 8. Exact training configuration as executed
**Executed Command:**
```bash
/home/harshil/.venvs/global/bin/python3 -u -m model.train \
  --datasets RTM MIDV500 \
  --epochs 50 \
  --stage-name stage2run3 \
  --init-weights model/checkpoints/stage1_mvss_lite_ep45.pt \
  --use-balanced-sampler \
  --lr-schedule cosine \
  --early-stopping-patience 8 \
  --domain-adversarial
```

**Training Details Resolved:**
- **Optimizer:** `Adam`
- **Learning Rate:** Cosine Annealing schedule starting from `0.000100` (decayed down to `0.000074` by Epoch 18 halt).
- **Early Stopping Patience:** 8 epochs
- **GRL Lambda:** Ramped up smoothly over 50 epochs via the schedule: `lambda_p = (2 / (1 + exp(-10 * p))) - 1`
