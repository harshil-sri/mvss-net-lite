Before either pilot — carve out a held-out specificity probe. Pull ~15-20 dense-text authentic images from the MIDV500/RTM pool (include your original 3), and explicitly exclude them from the training split. Use this same fixed set at multiple checkpoints through each pilot, not just once at the end — a single endpoint reading can't tell you if specificity is stable, drifting, or degrading with exposure.

Stage 1 pilot — CASIAv2 + DEFACTO:

Real dataloader, shuffled, full SyncedAugmentations on.
pos_weight: seg=39.49, edge=817.24 (already scanned).
Run 1 full epoch. 
Check: loss curves smooth and descending, no NaN/oscillation. That's the bar here.

Stage 2 pilot — RTM + MIDV500 (the one that matters):

Real dataloader, shuffled across the full combined pool, augmentation fully on.
pos_weight: seg=230.63, edge=7766.23.
Run this independently from fresh ImageNet init, not chained after the Stage 1 pilot checkpoint — you want to isolate whether Stage 2's specific risk factors (extreme weight + augmentation + MIDV500 dilution) are safe, without conflating it with an undertrained Stage 1 checkpoint. Same isolation logic you've been using throughout this thread.
Target at least 2 full epochs over the ~24,000-image combined pool (~48,000 image-passes). 
At ~25%, ~50%, and 100% through the pilot: run the held-out specificity probe and log max-probability/threshold-crossings per image, so you can see a trend instead of one snapshot.
At the end: run a noise-branch ablation — forward pass with the Bayar branch zeroed vs. intact on a handful of held-out batches, compare loss/predictions. This is the actual test for Hypothesis D. If disabling the noise branch barely changes anything after real augmentation exposure, augmentation degraded its signal and it's not contributing — worth knowing before you commit days to the full run either way.