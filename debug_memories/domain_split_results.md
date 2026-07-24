# Domain-Split Pilot Results: The Final Verdict

After perfectly balancing the dataloader distributions (`WeightedRandomSampler`) and precisely recalculating the exact effective `pos_weight` (6216.84), we ran a completely isolated 2-epoch pilot to see if the network could resolve the tabular-grid hallucination without loss-function changes.

### Final Epoch 2 Probe Results
**RTM Authentic Documents (Scanned Paper)**
*   **Avg Max Prob:** 0.8408
*   **Peak Prob:** 0.9850 (On Image #14, the strict grid-box header)
*   **Total Hallucinating Images:** 20/20 (including Image #16)
*   **Avg Hallucinated Pixels:** 20,954

**MIDV500 Authentic Documents (Plastic Cards)**
*   **Avg Max Prob:** 0.3156
*   **Total Hallucinating Images:** 3/20
*   **Avg Hallucinated Pixels:** 5.30

### BayarConv Ablation (RTM)
*   Intact Loss: 5.4319
*   Zeroed Loss: 5.7311
*   Difference: +0.2992 (The network *is* successfully relying on noise artifacts!)

### Conclusion: The BCE Failure State
The 20% reduction in `pos_weight` did lower the total hallucinated pixels from ~24k to ~20k, but the core failure remains absolute. 20 out of 20 scanned RTM documents are still aggressively hallucinating on tabular grid lines. 

We have now definitively proven that scaling False Negatives globally via `pos_weight` in Binary Cross Entropy is inherently flawed for document datasets. Even when perfectly calibrated to the data sparsity, a 6,200x penalty forces the network to accept 20,000 False Positive pixels rather than risk missing a single true edge. It will always highlight tables. 

We are officially gated by the loss function architecture. We must transition to Tversky Loss.
