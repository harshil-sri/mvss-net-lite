# Stage 1 Checkpoint Comparison: Epoch 40 vs Epoch 45

**Methodology:**
- **Environment:** Executed strictly on CPU to ensure zero interference with the running Stage 2 GPU training job.
- **Probe Set:** 600 images explicitly drawn from the `manifest.json` `val` split (200 CASIAv2-authentic, 200 CASIAv2-forged, 200 DEFACTO-forged). This provides a significantly larger, more statistically robust sample size than the original 50-image set.

## Paired Probe Results (600 Images)

### Threshold 0.9
| Metric | Epoch 40 | Epoch 45 | Delta |
|--------|----------|----------|-------|
| CASIAv2 Authentic FP | 92 / 200 | 63 / 200 | **-29 (Improved)** |
| CASIAv2 Forged TP | 181 / 200 | 176 / 200 | -5 (Minor Drop) |
| DEFACTO Forged TP | 197 / 200 | 200 / 200 | **+3 (Improved)** |

**Arithmetic Reconciliation (Threshold 0.9):**
- Epoch 40 starting False Positives: **92**
- Images fixed (FP at 40 -> Correct at 45): **-42**
- Images regressed (Correct at 40 -> FP at 45): **+13**
- *Expected Epoch 45 FP (92 - 42 + 13):* **63** (Matches exactly)

### Threshold 0.5
| Metric | Epoch 40 | Epoch 45 | Delta |
|--------|----------|----------|-------|
| CASIAv2 Authentic FP | 159 / 200 | 133 / 200 | **-26 (Improved)** |
| CASIAv2 Forged TP | (Unrecorded)* | 194 / 200 | N/A |
| DEFACTO Forged TP | (Unrecorded)* | 200 / 200 | N/A |

*\*Note: TP at 0.5 for Epoch 40 was truncated in the original output buffer, but FP was precisely extracted.*

**Arithmetic Reconciliation (Threshold 0.5):**
- Epoch 40 starting False Positives: **159**
- Images fixed (FP at 40 -> Correct at 45): **-35**
- Images regressed (Correct at 40 -> FP at 45): **+9**
- *Expected Epoch 45 FP (159 - 35 + 9):* **133** (Matches exactly)

### Loss History (From Training Logs)
| Metric | Epoch 40 | Epoch 45 | Train/Val Gap |
|--------|----------|----------|---------------|
| Train Total Loss | 2.562 | 2.504 | |
| Val Total Loss | **3.419** | 3.491 | |
| **Gap** | **0.857** | 0.987 | (Epoch 40 tighter) |

---

## Verdict: The Tradeoff

Choosing between Epoch 40 and Epoch 45 represents a **genuine tradeoff** between global validation loss and domain-specific false-positive suppression.

**Why Epoch 40 is better for general validation:**
Epoch 40 has the objectively lower validation loss (3.419 vs 3.491) and a smaller train/val gap (0.857 vs 0.987). The divergence of the train and val loss curves begins just after Epoch 40, meaning Epoch 45 has technically begun the early stages of overfitting to the training distribution. This was the exact same metric used to disqualify Epoch 50 (which had a worse gap of 1.165).

**Why Epoch 45 is better for the probe task:**
Despite the slightly degraded general validation loss, the targeted domain-specificity probe proves that Epoch 45 is decisively better at suppressing False Positives on authentic images. The paired comparison on 200 images confirms this is a structural, non-random improvement: at the 0.9 threshold, 42 specific images were fixed while only 13 regressed (a net drop of ~31% FPs). This holds true at the 0.5 threshold as well (35 fixed vs 9 regressed).

**Conclusion:** 
Because the fundamental goal of this model iteration is to eliminate catastrophic hallucination on authentic document backgrounds, the direct measurement of False Positives on the probe set is given heavier weight than the global BCE loss (which is heavily diluted by background pixels). The robust, statistically significant reduction in False Positives makes **Epoch 45** the preferred anchor for Stage 2, accepting the slightly wider train/val gap as an acceptable tradeoff for superior hallucination suppression.
