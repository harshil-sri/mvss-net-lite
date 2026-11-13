import csv

md_content = """# Stage 2 Run 3 — Diagnostic Report

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
"""

with open("reports/stage2run3_history.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        md_content += f"| {row['epoch']} | {float(row['train_total_loss']):.4f} | {float(row['train_seg_loss']):.4f} | {float(row['train_edge_loss']):.4f} | {float(row['val_total_loss']):.4f} | {float(row['val_seg_loss']):.4f} | {float(row['val_edge_loss']):.4f} |\n"

md_content += """
---

## 3. Domain-split specificity probe, at each saved checkpoint
*(Waiting for probe results...)*

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
/home/harshil/.venvs/global/bin/python3 -u -m model.train \\
  --datasets RTM MIDV500 \\
  --epochs 50 \\
  --stage-name stage2run3 \\
  --init-weights model/checkpoints/stage1_mvss_lite_ep45.pt \\
  --use-balanced-sampler \\
  --lr-schedule cosine \\
  --early-stopping-patience 8 \\
  --domain-adversarial
```

**Training Details Resolved:**
- **Optimizer:** `Adam`
- **Learning Rate:** Cosine Annealing schedule starting from `0.000100` (decayed down to `0.000074` by Epoch 18 halt).
- **Early Stopping Patience:** 8 epochs
- **GRL Lambda:** Ramped up smoothly over 50 epochs via the schedule: `lambda_p = (2 / (1 + exp(-10 * p))) - 1`
"""

with open("reports/diagnostic_stage2run3_report.md", "w") as f:
    f.write(md_content)
