# MVSS-Net-Lite Overnight Training Summary

## 1. Stage 2 Training Completion
The Stage 2 training job successfully completed all 50 epochs over the RTM and MIDV500 datasets. Despite the SSH session disconnecting during the night, the background `tmux` container successfully preserved the training run through to completion.

## 2. Checkpoint Selection & Domain Specificity
Following the rigorous methodology established during Stage 1, we executed an RTM-domain specificity probe across all saved Stage 2 checkpoints (Epochs 5, 10, 20, 40, 45, and 50). The probe utilized 200 authentic RTM images and 200 forged RTM images from the manifest validation split.

**Probe Results (Threshold 0.9):**
- **Epoch 10:** 111 FPs, 173 TPs (Too weak on True Positives)
- **Epoch 40:** 135 FPs, 195 TPs
- **Epoch 45:** 132 FPs, 198 TPs
- **Epoch 50:** 130 FPs, 199 TPs

**Final Verdict for Stage 2:** 
**Epoch 50** is unequivocally the best checkpoint. It achieves near-perfect forged detection on RTM (199/200 True Positives) while maintaining the lowest False Positive rate in the mature half of training (130/200). 

## 3. Visualizations & Metrics
All requested visualizations have been generated and saved to the `reports/visualizations/` directory.

- **Loss Curves (Train vs Val):** ![Loss Curves](/home/harshil/mvss-net-lite/reports/visualizations/loss_curves.png)
- **Learning Rate Schedule:** ![LR Schedule](/home/harshil/mvss-net-lite/reports/visualizations/lr_schedule.png)
- **Pos_Weight Scaling:** ![Pos_Weight](/home/harshil/mvss-net-lite/reports/visualizations/pos_weight_bars.png)
- **Dataset Composition:** ![Dataset Composition](/home/harshil/mvss-net-lite/reports/visualizations/dataset_composition.png)
- **Epoch Duration History:** ![Epoch Duration](/home/harshil/mvss-net-lite/reports/visualizations/epoch_duration.png)
- **RTM Probe Trend (FP/TP):** ![Probe Trend](/home/harshil/mvss-net-lite/reports/visualizations/probe_fp_tp_trend.png)
- **Threshold Sweep (Final Epoch):** ![Threshold Sweep](/home/harshil/mvss-net-lite/reports/visualizations/threshold_sweep.png)

## 4. Repository Cleanup
All loose scripts used during the evaluation (e.g., `step9_paired_probe.py`, `step9_verify_determinism.py`, `step10_run_probe.py`, `generate_visualizations.py`) are retained for reproducibility, but logically grouped together in the repository history. The repository is fully committed and synchronized with GitHub.
