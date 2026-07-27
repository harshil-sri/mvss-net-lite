Here's the merged, sequential handoff — manifest build first (it blocks everything else), then Stage 1's rerun on top of it.

Task: Build a single unified manifest across all four datasets, wire it into the dataloader, then run Stage 1 (CASIAv2 + DEFACTO) for real on the corrected pipeline. Do not skip or reorder steps.

STEP 1 — Build the manifest
Write a one-time script that scans the full set of raw data directories for all four datasets used anywhere in this project — CASIAv2, DEFACTO, RTM, MIDV500 — as a single unified list of (image_path, mask_path, dataset_name) entries. Do not scan per-dataset; one combined list, one split.

Shuffle with a fixed seed and split 80/10/10 into train/val/test.
Save as manifest.json: three lists (train, val, test), each entry containing image path, mask path, and dataset tag.
Report total counts per split, and per-dataset counts within each split (so it's visible each dataset is represented across train/val/test, not accidentally dropped from one).

STEP 2 — Rewire get_dataloader
In data_pipeline/dataset_loader.py, rewrite get_dataloader(dataset_names, ...) to:

Load manifest.json once.
Filter the requested split (train/val/test) down to only entries matching dataset_names.
Build the Dataset/DataLoader from those fixed paths.
Remove the random_split call entirely from this path — no dataset should ever be split at request time again, only filtered from the pre-built manifest.
Leave get_splits() alone only if nothing else still depends on it; if something does, redirect it to the manifest too rather than leaving two competing split mechanisms in the codebase.

STEP 3 — Verify no leakage, cheaply
Run a quick check confirming zero overlap between manifest.json's train/val/test lists (trivial by construction, but confirm the file-writing/loading logic didn't introduce a bug — e.g. duplicate paths across splits). Report the result.

STEP 4 — Confirm CASIAv2 + DEFACTO are reachable through the new pipeline
Call get_dataloader(['CASIAv2', 'DEFACTO'], is_train=True, return_splits=True) once and confirm it returns non-empty train/val/test loaders pulling only from the manifest's CASIAv2/DEFACTO entries. Report the batch counts.

STEP 5 — Smoke test Stage 1
python -m model.train --datasets CASIAv2 DEFACTO --smoke-test --stage-name stage1. Confirm it runs end to end with no errors on the new manifest-based pipeline.

STEP 6 — Overfit sanity check
python -m model.train --datasets CASIAv2 DEFACTO --overfit-batch --stage-name stage1_overfit. Confirm loss crashes toward near-zero (not stuck ~2.7-2.8) — this re-verifies the ImageNet-pretrained backbone is actually working for this dataset combination, same check that originally caught the random-init bug.

STEP 7 — Full Stage 1 training
If Step 6 passes: python -m model.train --datasets CASIAv2 DEFACTO --epochs 50 --stage-name stage1. Let pos_weight auto-scan as the script already does. Report the scanned seg/edge pos_weight values.

STEP 8 — Domain-split specificity probe
Once checkpoints exist at epoch 10-15, run a CASIAv2-vs-DEFACTO specificity probe (same design as the earlier RTM-vs-MIDV500 one): check whether authentic CASIAv2 images get falsely flagged as forged, and whether that error rate looks suspiciously different from CASIAv2-forged detection accuracy. This checks whether the model is using dataset-domain as a shortcut (DEFACTO being 100% forged makes this the same risk category as the earlier MIDV500 bug).

Report the results.

STEP 9 — Stop and report
Do not chain into Stage 2 (--init-weights / --resume) yet. Report all results from Steps 1-8 and wait for review before deciding whether to restart Stage 2 from this Stage 1 checkpoint.

Explicitly out of scope for this handoff: building a CASIAv2/DEFACTO balanced sampler preemptively (wait for Step 8's evidence first — don't build a fix for a problem not yet confirmed to exist), any Tversky tuning, and touching the Stage 2 manifest-fix run already in progress — these run independently.