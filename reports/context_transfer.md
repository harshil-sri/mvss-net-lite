# Context Transfer: MVSS-Net-Lite Training

## Overall Goal
The user is running a 50-epoch overnight training job for `MVSS-Net-Lite` using a multi-stage process. 

## Completed Milestones
1. **Stage 1 (Epoch 1-50):** Successfully finished. Checkpoint `stage1_mvss_lite_ep45.pt` was selected based on optimal False Positive/True Positive ratios across CASIAv2 and DEFACTO datasets (lowest FP of 18/50 while maintaining 50/50 TP on DEFACTO).
2. **Data Loader Bug Fix:** Resolved a critical hang on WSL2 caused by CUDA multiprocessing by hardcoding `num_workers=0` in `dataset_loader.py`.
3. **Disk Space Recovery & Guardrails:** Deleted 170GB+ of bloated old checkpoints. Implemented strict guardrails in `model/train.py` to keep only the 3 most recent checkpoints (plus the absolute best one) and added a proactive `shutil.disk_usage` check that will cleanly halt training if free space drops below 15GB.
4. **Tmux Migration:** The training was safely migrated to a detached `tmux` session named `stage2_training` to prevent SSH connection drops from killing the run.

## Current Live Status
- **Stage 2 Training** is actively running inside the detached `tmux` session (`stage2_training`).
- It is trained on the dataset split: `RTM` and `MIDV500` with the `WeightedRandomSampler` active.
- As of the latest check, it had completed **Epoch 21/50**.
- The training loss is actively converging (~3.23) and validation loss is holding stable (~3.33).

## How to Monitor the Run
Do NOT kill the tmux session unless instructed! You can monitor the progress passively by:
1. Reading the CSV: `view_file reports/stage2_history.csv`
2. Reading the live terminal output: `tail -n 50 reports/tmux_stage2.log` (Note: Because of bash redirection buffering, the terminal log only updates in large chunks. The CSV file is much more accurate and updates immediately at the end of every epoch).

## Remaining Tasks for the Overnight Job
Once Stage 2 naturally completes (reaches Epoch 50):
1. **Checkpoint Selection (Stage 2):** Run `step8_probe.py` on the Stage 2 checkpoints to verify if FP rates improved without breaking TP.
2. **Visualizations:** Generate requested PNGs (loss curves, LR, FP/TP probes) for the final report.
3. **Repository Reorganization:** Move all loose files, scratch scripts, and old diagnostics into the `debug_memories/` folder.
4. **Git Push:** Ensure `.gitignore` ignores checkpoints/datasets, and push the final work to GitHub.

## Previous Conversation ID
If you need to query transcripts for exact reasoning: `cb72f17a-b058-447a-ba16-3c09ee566973`
