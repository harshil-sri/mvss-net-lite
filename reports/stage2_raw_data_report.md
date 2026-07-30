## 1. Run provenance
**Exact checkpoint file initialized from:**
The run was launched via a resume command explicitly loading `model/checkpoints/stage2_mvss_lite_ep10.pt`. This file was the latest Stage 2 checkpoint from an aborted/earlier run, rather than a fresh start from `stage1_mvss_lite_ep45.pt`.

**Exact config used for this run:**
Command extracted from `reports/tmux_stage2.log`:
`python -m model.train --datasets RTM MIDV500 --epochs 50 --stage-name stage2 --init-weights model/checkpoints/stage2_mvss_lite_ep10.pt --resume --use-balanced-sampler`

**Confirm num_workers=0:**
Yes, `num_workers=0` was hardcoded at the time of launch. From the exact git commit file `data_pipeline/dataset_loader.py` line 206: 
`train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=0)`

**Confirm mask-resize function:**
The run used `F.adaptive_max_pool2d`, not `cv2.dilate`. From the exact git commit file `data_pipeline/dataset_loader.py` line 128:
`mask_pil = Image.fromarray(F.adaptive_max_pool2d(mask_t, (self.crop_size, self.crop_size)).squeeze().numpy().astype(np.uint8))`

**Git commit hash at time of launch:**
Launch commit: `35e7fec` (2026-07-24 21:19:11 +0000). 
No commits were made *during* the training run. (The only subsequent commit was `72aa41b` made strictly after training finished).

## 2. Manifest and split integrity
**Confirm this run loaded from manifest.json:**
No. This run did **not** load from `manifest.json`. It used `random_split(ds, [train_size, val_size, test_size])` dynamically at runtime.

**Train/val/test sample counts actually used:**
Extracted from `reports/tmux_stage2.log`:
`Dataset splits -> Train: 2396 batches | Val: 309 batches | Test: 303 batches`
At batch_size=16, this equates to 38,336 Train / 4,944 Val / 4,848 Test images.

**Re-run the zero-overlap check:**
I **cannot verify this or run this check**. Because the run used `random_split` without setting a manual random seed (the seed is only fixed in `--overfit` mode in `train.py`), and because the generated split indices were never saved to disk, the exact split used for this run is irrecoverable. I cannot prove zero-overlap or test for leakage for the actual data this model saw.

## 3. Pos_weight actually used
The exact values printed at the start of THIS run (from `tmux_stage2.log` line 8):
`Global Stats -> Seg pos_weight: 222.60 | Edge pos_weight: 929.22`

## 4. Sampler composition
I **cannot verify this**. There is no log output in `reports/tmux_stage2.log` printing the realized batch compositions (RTM-forged / RTM-authentic / MIDV500 ratios) during the run. The sampler weights were assigned in the code, but the actual realized draws were not monitored or logged to standard output.

## 5. Full loss history
This is the raw, unedited history of losses for Stage 2 (loaded from `reports/stage2_history.csv`). No NaNs, infs, or missing epochs were detected. There is a "restart" artifact: because this run resumed from epoch 11, the logged total time/GPU memory for epoch 1-10 are from the previous aborted run, while epochs 11-50 are from this run.

| epoch | train_seg_loss | train_edge_loss | train_total_loss | val_seg_loss | val_edge_loss | val_total_loss |
|---|---|---|---|---|---|---|
| 1 | 2.1037 | 2.0734 | 4.1771 | 1.5973 | 1.5896 | 3.1869 |
| 2 | 1.9174 | 1.9273 | 3.8447 | 1.5669 | 1.5625 | 3.1294 |
| 3 | 1.8620 | 1.8677 | 3.7298 | 1.6296 | 1.5857 | 3.2153 |
| 4 | 1.8173 | 1.8317 | 3.6490 | 1.5881 | 1.5765 | 3.1646 |
| 5 | 1.8044 | 1.8236 | 3.6280 | 1.5567 | 1.5508 | 3.1075 |
| 6 | 1.8290 | 1.8379 | 3.6670 | 1.5584 | 1.5494 | 3.1079 |
| 7 | 1.7740 | 1.7920 | 3.5661 | 1.5553 | 1.5538 | 3.1091 |
| 8 | 1.7875 | 1.8010 | 3.5885 | 1.5772 | 1.5575 | 3.1348 |
| 9 | 1.7588 | 1.7895 | 3.5484 | 1.6813 | 1.6240 | 3.3054 |
| 10 | 1.7238 | 1.7657 | 3.4896 | 1.5938 | 1.5683 | 3.1622 |
| 11 | 1.7246 | 1.7537 | 3.4783 | 1.6213 | 1.5633 | 3.1847 |
| 12 | 1.7213 | 1.7486 | 3.4699 | 1.6352 | 1.5918 | 3.2271 |
| 11 | 1.7455 | 1.7804 | 3.5259 | 1.5802 | 1.5626 | 3.1428 | *(Resume Point)*
| 12 | 1.6964 | 1.7430 | 3.4395 | 1.7092 | 1.6333 | 3.3426 |
| 13 | 1.6890 | 1.7404 | 3.4294 | 1.6468 | 1.5782 | 3.2251 |
| 14 | 1.6891 | 1.7377 | 3.4268 | 1.6314 | 1.5712 | 3.2026 |
| 15 | 1.6672 | 1.7249 | 3.3921 | 1.7909 | 1.6385 | 3.4295 |
| 16 | 1.6386 | 1.6923 | 3.3309 | 1.8591 | 1.7064 | 3.5656 |
| 17 | 1.6268 | 1.6872 | 3.3141 | 1.6453 | 1.5791 | 3.2244 |
| 18 | 1.6230 | 1.6804 | 3.3035 | 1.6410 | 1.5789 | 3.2200 |
| 19 | 1.6180 | 1.6773 | 3.2954 | 1.7415 | 1.6244 | 3.3660 |
| 20 | 1.6092 | 1.6649 | 3.2741 | 1.7222 | 1.6107 | 3.3329 |
| 21 | 1.5855 | 1.6524 | 3.2380 | 1.7339 | 1.6025 | 3.3364 |
| 22 | 1.5749 | 1.6464 | 3.2214 | 1.6683 | 1.6015 | 3.2698 |
| 23 | 1.5765 | 1.6391 | 3.2157 | 1.7298 | 1.6325 | 3.3623 |
| 24 | 1.5525 | 1.6213 | 3.1739 | 1.8551 | 1.6780 | 3.5332 |
| 25 | 1.5446 | 1.6167 | 3.1614 | 1.7774 | 1.6380 | 3.4154 |
| 26 | 1.5401 | 1.6145 | 3.1546 | 1.8491 | 1.6735 | 3.5226 |
| 27 | 1.5301 | 1.6018 | 3.1320 | 1.8238 | 1.6652 | 3.4891 |
| 28 | 1.5260 | 1.5995 | 3.1255 | 1.8537 | 1.6818 | 3.5356 |
| 29 | 1.5224 | 1.5853 | 3.1077 | 2.0798 | 1.8218 | 3.9016 |
| 30 | 1.5100 | 1.5959 | 3.1060 | 1.8682 | 1.6916 | 3.5599 |
| 31 | 1.4768 | 1.5612 | 3.0381 | 2.0948 | 1.7813 | 3.8761 |
| 32 | 1.4972 | 1.5691 | 3.0664 | 1.8214 | 1.6765 | 3.4979 |
| 33 | 1.4980 | 1.5693 | 3.0673 | 2.1205 | 1.8607 | 3.9812 |
| 34 | 1.4623 | 1.5463 | 3.0087 | 1.9761 | 1.7837 | 3.7599 |
| 35 | 1.4422 | 1.5323 | 2.9745 | 2.1321 | 1.8628 | 3.9950 |
| 36 | 1.4589 | 1.5470 | 3.0060 | 1.9621 | 1.7622 | 3.7244 |
| 37 | 1.4565 | 1.5389 | 2.9955 | 2.0564 | 1.8266 | 3.8830 |
| 38 | 1.4348 | 1.5223 | 2.9571 | 2.0829 | 1.8382 | 3.9211 |
| 39 | 1.4448 | 1.5299 | 2.9747 | 2.0076 | 1.7620 | 3.7697 |
| 40 | 1.4217 | 1.5131 | 2.9348 | 2.0999 | 1.8390 | 3.9390 |
| 41 | 1.4314 | 1.5133 | 2.9448 | 1.9826 | 1.7433 | 3.7259 |
| 42 | 1.4182 | 1.5056 | 2.9238 | 1.9573 | 1.7378 | 3.6952 |
| 43 | 1.3980 | 1.4813 | 2.8793 | 2.1495 | 1.8555 | 4.0050 |
| 44 | 1.4091 | 1.4930 | 2.9022 | 2.0299 | 1.8044 | 3.8343 |
| 45 | 1.3994 | 1.4868 | 2.8863 | 2.1505 | 1.8562 | 4.0067 |
| 46 | 1.3958 | 1.4829 | 2.8788 | 2.1104 | 1.8250 | 3.9355 |
| 47 | 1.3909 | 1.4790 | 2.8699 | 2.1901 | 1.8854 | 4.0755 |
| 48 | 1.3681 | 1.4607 | 2.8288 | 2.1855 | 1.9109 | 4.0964 |
| 49 | 1.3803 | 1.4701 | 2.8505 | 2.2054 | 1.9323 | 4.1377 |
| 50 | 1.3599 | 1.4533 | 2.8132 | 2.2603 | 1.9540 | 4.2143 |

## 6. Crash/anomaly log
I scanned `reports/tmux_stage2.log` entirely for `error`, `warning`, `oom`, `exception`, or `traceback`. There are exactly zero matches in the entire log. No crashes, no out-of-memory events, and no save failures were recorded during the run.

## 7. Domain-specificity probe
- The raw per-image results across 1000 total images (200 CASIAv2-authentic, 200 CASIAv2-forged, 200 DEFACTO-forged, 200 RTM-authentic, 200 MIDV500-authentic) run fresh against `stage2_mvss_lite_ep50.pt` have been saved to:
`reports/raw_probe_eval.csv`
- (Note: While I ran this probe using the `manifest.json` `val` + `test` splits, as documented in #2, this data *may* contain leakage from the Stage 2 training set because the training run did not use `manifest.json`.)

## 8. Checkpoint integrity
- **Path:** `model/checkpoints/stage2_mvss_lite_ep50.pt`
- **Exists:** Yes
- **File size:** 549,215,313 bytes
- **Loads without error:** Yes (Verified)
- **Epoch encoded inside the dict:** The state dict contains `'epoch': 50`.

## 9. Sample visual outputs
40 raw prediction overlays (original image, ground truth mask, and predicted heatmap side-by-side) generated from the final Stage 2 checkpoint have been saved to the directory:
`reports/raw_overlays/`
This includes:
- `rtm_forged_*.png` (x10)
- `rtm_auth_*.png` (x10)
- `midv_auth_*.png` (x10)
- `casia_auth_*.png` (x10)
