Task: Fully autonomous overnight run. Wait for Stage 1 to finish, select the best checkpoint, launch Stage 2 fine-tuning, manage storage responsibly, and produce a full visual report — all without waiting for confirmation. If anything goes wrong or looks abnormal, STOP, save what you have, and write a clear explanation at the top of the summary — do not guess a fix and keep going unsupervised.

STEP 1 — Wait for Stage 1 to complete
Monitor the running 50-epoch Stage 1 job. Do not proceed until it finishes or fails. If it fails, stop immediately, log the error clearly, and do not attempt Stage 2.

STEP 2 — Evaluate every saved Stage 1 checkpoint
For every checkpoint saved during the run (not just 15/30/50 — whichever interval was actually used), run the domain-specificity probe (CASIAv2-authentic FP, CASIAv2-forged TP, DEFACTO-forged TP, threshold 0.9) plus pull the logged train/val loss for that epoch. Save this as a table.

STEP 3 — Select the best checkpoint, with a written justification
Selection rule: lowest FP-image-count with TP recall (both CASIAv2 and DEFACTO) not degraded relative to its peak. If multiple checkpoints tie, prefer the later one unless train/val loss shows clear divergence (val flat or rising while train keeps falling) at that point, in which case prefer the earlier one. Write the selected checkpoint, the numbers behind the decision, and the reasoning to reports/stage1_checkpoint_selection.md. Do not skip this — it needs to be reviewable in the morning, not just acted on.

STEP 4 — Launch Stage 2, chained from the selected checkpoint
Use --init-weights <selected_checkpoint> --resume (or equivalent) into the existing Stage 2 config: manifest-based pipeline, corrected mask-resize, WeightedRandomSampler (RTM-forged/RTM-authentic/MIDV500), and pos_weight re-scanned live at run start (do not hardcode 1058.38 — the dataset composition changed from init weights, but the scan logic already handles this automatically). Do not enable Tversky tuning or hard-negative mining — out of scope for this run.

STEP 5 — Checkpoint and log discipline (disk safety)

Save a checkpoint every 5 epochs, not every epoch.
Keep only the 3 most recent checkpoints plus the single best-so-far (by val loss or probe FP rate) at any time — delete older ones automatically as new ones are saved.
Before every checkpoint write, check free disk space (shutil.disk_usage or equivalent). If free space drops below 15GB, stop training immediately, preserve the most recent good checkpoint, and write a clear warning to the top of the final report. Do not let this repeat the disk-full crash from earlier tonight.
Log loss/metrics every epoch to a lightweight text/CSV log (cheap), even though checkpoints are saved less often.

STEP 6 — Run the same domain-specificity probe on Stage 2 checkpoints
At each saved Stage 2 checkpoint, run the RTM-domain specificity probe (authentic RTM FP, forged RTM TP, threshold sweep 0.5-0.9) on a probe set pulled from the manifest's val split — confirm zero overlap with train before using it, the same way it was verified for Stage 1.

STEP 7 — Generate visualizations (matplotlib, saved as PNGs to reports/visualizations/)
Produce all of the following, each as its own saved PNG:

Loss curves — total/seg/edge, train vs val, per epoch, for both Stage 1 and Stage 2.
Learning rate schedule over epochs (cosine annealing curve).
pos_weight values used per stage/head (simple bar chart).
TP and FP pixel counts across epochs from the probe results (line chart, both stages).
Hallucinating-image-count across epochs at threshold 0.9 (line chart).
Threshold sweep curve (0.5 → 0.9) for the final Stage 2 checkpoint — FP count and TP count both plotted against threshold.
Confidence histograms — distribution of predicted probability on true-positive vs false-positive pixels, for the final checkpoint.
Sample prediction overlays — a grid of 6-8 example images (original / ground-truth mask / predicted heatmap side by side), including at least 2 known-hard cases (e.g. dense tabular authentic images) and 2 clean successes.
Dataset composition chart — image counts per dataset (CASIAv2/DEFACTO/RTM/MIDV500) and per split (train/val/test), from the manifest.
Realized sampler batch composition (bar chart, actual vs target ratio) for Stage 2.
Epoch duration / training time bar chart.
One combined summary dashboard figure (multi-subplot) suitable for a single presentation slide, pulling the headline items from above (loss curves, FP/TP trend, threshold sweep, dataset composition).

STEP 8 — Write a top-level summary
Generate reports/overnight_summary.md linking every saved visualization and checkpoint, with a short plain-English paragraph per figure. If Step 5's disk guardrail ever triggered, or anything failed, this must be the first thing stated at the top of the file, in plain language, before anything else.

STEP 9 — Stop
Do not proceed to Tversky tuning, hard-negative mining, or any further pipeline changes. Stage 2 checkpoint + visual report is the full scope of tonight's task.

**STEP 10 — Reorganize, commit, and push to GitHub**

1. Do not delete any existing logs, reports, or debug scripts — even small ones. Instead, create a `debug_memories/` folder at the repo root and move anything loose/one-off there: old probe scripts, standalone diagnostic scripts (`check_overlap.py`, `check_recall.py`, synthetic mask-line tests, etc.), old text/CSV logs, and any prior report markdowns not already in `reports/`. Preserve original filenames so history stays traceable.

2. Lightly reorganize the active repo structure — group files sensibly (e.g. `data_pipeline/`, `model/`, `reports/`, `reports/visualizations/`, `debug_memories/`) without renaming or moving anything that active scripts import from, unless you also update those import paths. Do not do a deep refactor — this is tidying, not a rewrite.

3. Write/update `.gitignore` to exclude: all model checkpoints (`*.pt`, `model/checkpoints/`), the manifest's referenced raw image/mask data directories, any cache/tmp folders, `__pycache__/`, virtual environments, and any single file over ~50MB. Do **not** gitignore anything in `debug_memories/`, `reports/`, or the visualization PNGs — those are small and meant to be kept and visible.

4. Before committing, run `git status` and manually scan for anything large that isn't yet gitignored (`du -sh` on staged files or `git diff --stat`) — confirm nothing checkpoint-sized or dataset-sized is about to be committed. If something large slips through, add it to `.gitignore` and unstage it before proceeding.

5. Commit in multiple small, logical, human-sounding commits reflecting the actual work done — not one giant dump and not robotic messages like "update files" or "fix stuff." Group by what actually changed, e.g.:
   - `fix: replace random_split with unified manifest-based dataset splitting`
   - `fix: correct mask-resize interpolation (NEAREST -> block max pooling) to stop thin-edge erosion`
   - `feat: add domain-specificity probes for Stage 1 and Stage 2`
   - `feat: Stage 1 training with corrected ImageNet backbone init`
   - `fix: checkpoint saving now includes optimizer state and disk-space guard`
   - `chore: reorganize repo, move old debug scripts/logs into debug_memories/`
   - `docs: add training visualizations and overnight run summary`
   Adjust groupings to match whatever actually changed tonight — don't force this exact list if it doesn't match reality.

6. Confirm a GitHub remote (`git remote -v`) already exists before pushing. If none is set, **stop and do not create a new repo or guess a destination** — flag this in the final report instead of proceeding.

7. Push to the current branch. Do not force-push, do not merge into `main`/`master` if the current branch isn't already that branch — push the working branch as-is and leave any merge decision for review in the morning.

8. Report the final GitHub URL/branch and commit list at the top of `reports/overnight_summary.md`, alongside everything else from Step 8.