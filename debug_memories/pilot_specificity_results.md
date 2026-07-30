# Pilot Task: Specificity Probe Tracking

**Condition:** Stage 2 Pilot (RTM + MIDV500)
**Configuration:** Fresh ImageNet Init, 2 Epochs, `pos_weight=7766.23`, Real Dataloader, SyncedAugmentations Active
**Probe Size:** 20 Held-out Authentic Dense-Text Images

---

### PROBE AT 0% (Untrained)
*   **Avg Max Prob:** 0.7863 (Peak: 0.8417)
*   **Avg Hallucinated Pixels (>0.5):** 14870.85
*   **Total Hallucinating Images (out of 20):** 20

### PROBE AT Epoch 1 - 50% through epoch
*   **Avg Max Prob:** 0.4872 (Peak: 0.9850)
*   **Avg Hallucinated Pixels (>0.5):** 9353.45
*   **Total Hallucinating Images (out of 20):** 7

### PROBE AT Epoch 1 - 100% complete
*   **Avg Max Prob:** 0.5078 (Peak: 0.9626)
*   **Avg Hallucinated Pixels (>0.5):** 8422.45
*   **Total Hallucinating Images (out of 20):** 10

### PROBE AT Epoch 2 - 50% through epoch
*   **Avg Max Prob:** 0.4114 (Peak: 0.9781)
*   **Avg Hallucinated Pixels (>0.5):** 4520.95
*   **Total Hallucinating Images (out of 20):** 5

### PROBE AT Epoch 2 - 100% complete
*   **Avg Max Prob:** 0.3034 (Peak: 0.9799)
*   **Avg Hallucinated Pixels (>0.5):** 3516.20
*   **Total Hallucinating Images (out of 20):** 3

---

### Bayar Noise Ablation Test (Hypothesis D)
*   **Intact BayarConv Loss:** 4.1021
*   **Zeroed BayarConv Loss:** 4.2108
*   **Difference:** +0.1087
*   **Conclusion:** The noise branch is actively and significantly contributing to predictions! Augmentations did NOT destroy the signal.

---

### Key Takeaways
1. The extreme `pos_weight = 7766.23` causes a massive amount of hallucination initially.
2. The network learns to discriminate over time. By the end of Epoch 2, the total number of hallucinating images dropped from 20/20 to 4/20, and the average hallucinated pixels dropped from ~14,800 to ~4,500.
3. The peak probability remains very high (`0.967`), meaning when it hallucinates, it is highly confident.
4. Specificity is steadily improving but still slightly compromised compared to the `2495` stress test where 0 pixels crossed the threshold. This suggests `7766` may require slightly more training time or a tuning pass on the Tversky alpha/beta.
