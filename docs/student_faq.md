# MVSS-Net Lite — Student FAQ

A plain-English tour of everything this project does and *why*. Written for a
deep-learning-for-CV student who wants to actually understand the architecture,
the losses, and the war stories hidden in this repo. Every number here comes
from real logs/tests in this repo — nothing is invented.

---

## Part 1 — The problem

### Q1. What are we building?

A model that looks at a **scanned document** and outputs a **pixel-level map of where it was tampered with** — spliced text, altered fields, forged signatures — plus an outline of suspicious edges. Not just "this document is fake with 87% confidence", but *"here is exactly which region, and here's my evidence."*

### Q2. Why segmentation instead of plain classification?

Classification tells you **what**; segmentation tells you **where**. For a forensic reviewer, "where" is the whole point — they have to defend the finding. Technically, we output a heatmap the same size as the input (`256×256`, same as the image crop) where each pixel gets a forgery probability. That's a dense prediction task, so the network is a U-Net-style encoder–decoder, not a classifier head.

### Q3. What is a "domain shortcut," and why did it almost sink Stage 2?

This is the most important ML lesson in the repo.

In Stage 2 we fine-tune on RTM (scanned documents, mostly forgeries) + MIDV500 (ID-card scans, 100% authentic). The model notices something lazy: *"RTM images look like scanned paperwork → say 'forged'. MIDV500 images look like ID cards → say 'authentic'."* Dataset identity becomes a proxy for the label. It's a **spurious correlation** — the classic shortcut-learning failure.

Run 1 paid for it: on authentic RTM scans the model hallucinated forgeries on **65% of images @ threshold 0.9 (76.5% @ 0.5)**, while authentic MIDV500 only triggered **2%/7%**. If your detector cries "fake!" on two-thirds of genuine paperwork, it's useless in production no matter how good the loss curve looks.

The counter-moves in this repo: balanced sampling so neither domain dominates a batch, and (the new experiment) freezing early layers so the general features from Stage 1 don't get dragged toward "scanned = forged."

---

## Part 2 — The architecture

### Q4. Why two branches?

Documents hide tampering in two different signals:

1. **Appearance/structure** — text layout, edges, shapes (RGB branch).
2. **Sensor/noise fingerprints** — every camera/scanner leaves a characteristic noise pattern; manipulated regions have *inconsistent* noise (noise branch).

Each branch is its own ResNet-34 (`model/backbone.py`: `edge_*` and `noise_*` modules). Fusion happens at four scales, so the final decision always combines both evidence types.

### Q5. What is a Bayar-constrained convolution?

(`model/constrained_conv.py`) A normal 3×3 or 5×5 conv can learn anything. The Bayar conv is **handcuffed by design**: its center weight is fixed at **−1**, and all surrounding weights must sum to **+1**.

Why? Think about what that kernel computes: `output = neighborhood_sum − center`. If a patch is locally smooth (pure noise texture, no structure), the output ≈ 0. Only when the center pixel deviates from its surround — i.e., **high-frequency residual/noise content** — does the output light up. It's a hard-wired high-pass filter that the network can still tune, but can never turn into something else. The noise branch starts from this constrained conv, then feeds ResNet stages.

### Q6. Why call `normalize_weights()` after every single optimizer step?

Because Adam doesn't respect our constraints. Gradient descent happily pushes weights wherever the loss wants — including breaking "center = −1, surround = 1". So after each step we **re-project** the kernels back onto the constraint surface (`train.py` calls `model.backbone.noise_extractor.normalize_weights()`). It's like projecting onto a feasible set in constrained optimization: take a step, snap back, repeat.

### Q7. What does CBAM fusion actually do? (`model/fusion.py`)

At each of the 4 ResNet stages, we have a noise feature map and an edge feature map. CBAM (Convolutional Block Attention Module) decides **what to keep** before fusing:

- **Channel attention** (global-average-pool → tiny MLP → sigmoid): "which feature *channels* matter right now?"
- **Spatial attention** (max+avg maps over channels → 7×7 conv → sigmoid): "which *locations* matter?"

Design decisions made here (documented in the file header):
- **Separate attention per branch** — noise and edges encode different information, so each gets its own channel/spatial weighting instead of sharing one.
- **Fusion via concat + 1×1 conv**, not element-wise addition — let the network *learn* the blend ratio instead of assuming both branches contribute equally everywhere.

### Q8. Why ResNet-34 ("Lite")?

The original MVSS-Net uses heavier backbones. We're on a single RTX 4060 (8GB) training on tens of thousands of images — ResNet-34 keeps the residual-learning benefits (skip connections ease gradient flow, BasicBlocks stack cheaply) while fitting memory/compute. Total model budget: **45.7M parameters** (measured).

### Q9. Why two output heads — segmentation AND edge?

The seg head localizes the forged region. The **edge head predicts the boundary** of that region (trained against Canny edges of the ground-truth mask). Two reasons:
1. **Multi-task regularization** — predicting boundaries forces features that respect shape, which improves seg quality.
2. **Evidence generation** — the demo/report layer uses edge maps as human-checkable proof.

Both heads run off the same decoded features; each gets its own loss (weights 1.0/1.0).

### Q10. How does the decoder work?

Classic U-Net flavor (`model/network.py`): take the deepest fused features (512ch, spatially 1/16), upsample, concatenate with the skip features from the level above, convolve; repeat down to 64ch at full resolution; then two heads produce seg + edge maps. Skip connections re-inject high-resolution detail that the deep layers lost — essential for pixel-precise masks.

---

## Part 3 — Losses

### Q11. Why BCE with `pos_weight`, and why is edge pos_weight ~800–2500?!

Forged pixels are vanishingly rare: a typical document has >99.9% background. Plain BCE would learn to say "background everywhere" and score well. `pos_weight` multiplies the positive-class term of BCE, effectively saying *"a missed forged pixel hurts N× more than a false alarm."*

N is measured live from the data at launch (`train.py` scans the whole train loader): latest smoke scan gave **Seg pos_weight 242.49, Edge pos_weight 795.19**; older runs measured up to ~2495. Edges are even rarer than forged regions (they're 1px-thin outlines), hence the bigger weight. Live re-scanning matters: Run 1 logged **222.60 / 929.22** under its (broken) split — if you hardcode stale values you silently mistune the trade-off.

### Q12. What's Tversky loss, and why α=0.3 / β=0.7?

Tversky generalizes Dice/IoU by penalizing false positives (FP) and false negatives (FN) asymmetrically:

```
T = intersection / (intersection + α·FP + β·FN)
loss = 1 − T
```

With **α=0.3, β=0.7** we punish missed forgery pixels 2.3× more than hallucinated ones. That matches the forensic cost function: better to flag an extra region for review than to clear a tampered field. (Note: tuning α/β further is deliberately deferred — single-variable discipline.)

### Q13. Why BCE **and** Tversky together?

BCE gives dense per-pixel gradients even where predictions are confidently wrong; region-overlap losses (Dice/Tversky) optimize the thing you actually measure (overlap) but have weak gradients when overlap ≈ 0. Their failure modes complement each other, so each head trains on `BCE(pos_weight) + Tversky/Dice`.

---

## Part 4 — Data pipeline war stories

### Q14. What exactly did `random_split` break?

`torch.utils.data.random_split` shuffles a dataset into splits **at runtime** using the global RNG. Run 1 called it without fixing a seed and never saved the indices. Result: **the exact train/val/test membership of Run 1 is unrecoverable forever**. Worse, the eval probe used `manifest.json` — a *different* random partition of the same pool — so an estimated ~80% of "held-out" probe images had probably been trained on. The scary FP numbers above were therefore a likely-*optimistic* floor. Unreproducible split ⇒ unverifiable evaluation ⇒ the run was discarded in full.

### Q15. What's a manifest split?

Compute the 80/10/10 split **once** over the unified pool (`model/scripts/build_manifest.py`, seed 42), save it to `reports/manifest.json`, and make every consumer filter that file. Splits become immutable data, not runtime accidents. Our integrity check proves: **54,805 total, zero pairwise image/mask overlap, zero intra-split duplicates, zero missing files.**

### Q16. Tell me about the mask-resize bug (the ~87% erasure).

Masks get downscaled from e.g. 1000px to the 256px crop. The old code used PIL `NEAREST` — for a **1-pixel-wide line**, nearest-neighbor sampling usually lands on background pixels, so the line simply vanishes. Measured on our synthetic probe: NEAREST kept **1 of 206** columns of an 800px horizontal line (~87%+ of thin structures destroyed elsewhere in testing). The fix: `F.adaptive_max_pool2d` — a pixel survives if ANY source pixel in its window survived. Probe result through today's live loader: **206/206 columns preserved, 806 px total, byte-identical behavior to the reference fix**. Moral: resizing *labels* needs max-pooling semantics, and "obviously safe" refactors deserve regression probes.

### Q17. How does WeightedRandomSampler force 1/3–1/3–1/3 batches? Did it work?

`WeightedRandomSampler(weights, replacement=True)` draws indices i.i.d. with P(i) ∝ w_i. To equalize classes, set w_i ∝ 1/(count of i's class). Then expected class mass is equal by construction.

We measured the REAL manifest train pool (mask-sum > 0 ⇒ forged): **rtm_forged 4,815 / rtm_authentic 2,360 / midv500 11,986**. Old code hardcoded denominators 6000/3000/15050 — wrong counts, but replaying those weights against today's pool lands within ~0.6pp of target anyway. The new dynamic scheme measures counts at load time and drew **33.28% / 33.39% / 33.33% over 95,805 simulated draws** (±0.06pp of target).

Two honest footnotes. First: this is a **reconstruction, not a measurement of Run 1** — Run 1's actual draws were never logged and its split is unrecoverable, so "the old sampler was probably fine" is an inference from code + today's data, not a verified fact about Run 1. Second: equal *ratios* don't mean equal *exposure*. Measured repetition per epoch: each drawn rtm_authentic image appears ~**2.9×** per epoch (max 12), vs ~1.3× for MIDV500 images; ~59% of MIDV500 isn't even seen in a given epoch. A 2,360-image authentic pool carrying a third of every batch means heavy repetition of RTM-scan appearance — a plausible secondary contributor to memorization that we've now quantified but not yet experimentally tested.

### Q18. Why `num_workers=0`?! Isn't that slow?

Yes it's slow. But on WSL2, PyTorch DataLoader workers fork the CUDA-initialized process → deadlock; `spawn` also crashes there. `num_workers=0` (data loading inline, no subprocesses) is the confirmed workaround, hardcoded in every active DataLoader construction and re-verified by grep. Don't "optimize" this without a new mitigation and an isolated test — it's burned this project once.

---

## Part 5 — Reading training curves

### Q19. How do you spot overfitting? What happened in Run 1?

Signature: **train loss keeps falling while val loss rises**. Run 1: val bottomed ~**3.11** around epochs 5–10, then climbed almost monotonically to **4.21** by epoch 50, while train fell **3.63 → 2.81** (final gap 1.40). The model wasn't learning forensics anymore; it was memorizing its training partition.

### Q20. Why did Stage 1 ship epoch 45 instead of epoch 50?

Because "last checkpoint" ≠ "best checkpoint." At ep50, CASIAv2-authentic FPs were **24/50** and val total loss had started rising (3.589) while train kept dropping (2.424) — early overfitting. Epoch 45 had the lowest FP count (**18/50**) with essentially peak recall (DEFACTO 50/50, CASIAv2 46/50). Lesson: select checkpoints at the **val-loss bottom**, cross-checked with domain-specific FP probes, not by schedule.

---

## Part 6 — This run's experiment: layer freezing

### Q21. What are we freezing and why?

Hypothesis: fully fine-tuning 45.7M params on ~24k low-diversity images gives the net too much freedom to memorize dataset identity. Freezing the early layers (already excellent generic feature extractors after Stage 1) cuts the effective hypothesis space.

| Component | Params | Status |
|---|---|---|
| Noise stem + L1 + L2 + Bayar conv | 1,348,129 | ❄️ frozen |
| RGB/edge stem + L1 + L2 | 1,347,904 | ❄️ frozen |
| Both branches L3 + L4 | 39,873,536 | 🔥 trainable |
| CBAM ×4 | 787,152 | 🔥 trainable |
| Decoder + both heads | 2,360,898 | 🔥 trainable |

Three implementation details students always miss:
1. **Optimizer over trainable params only** — `filter(requires_grad)`; otherwise Adam allocates state for tensors that never update.
2. **Frozen BatchNorm stays in `.eval()` mode** — BN running stats are updated by *forward passes*, not gradients. Leave them in train() and your "frozen" early layers keep adapting to the new domain's statistics — a half-frozen Frankenstein state that quietly degrades performance. We pin **32 BN layers** back to eval after every `model.train()` call.
3. **Fresh init via `--init-weights` from `stage1_mvss_lite_ep45.pt`, never `--resume` from Run 1** — Run 1's weights carry the shortcut; we want Stage 1's knowledge, not its habits.

### Q22. Why discard Run 1 entirely instead of continuing from it?

Two reasons: (a) its split is unrecoverable, so any metric derived from it is untrustworthy; (b) its weights encode the very shortcut we're trying to unlearn. Starting fresh from Stage 1 ep45 is the cleanest "single variable" baseline: same init as Run 1 *should* have had, minus its sins.

---

## Part 7 — Ops guardrails

### Q23. Checkpoint hygiene?

Unattended runs die stupid ways: disk fills mid-`torch.save` (corrupt checkpoint), or checkpoints pile up. Guardrails now: save every 5 epochs only; **check free space before every save** (halt gracefully + write `reports/DISK_FULL_WARNING.txt` below 15GB); rotate keeping last 3 but **never delete the current best-val checkpoint**. All unit-tested with mocked disk space — no epochs required.

### Q24. What's "single-variable discipline"?

If you change two things between runs and results move, you don't know which one did it. Each run changes exactly ONE experimental lever (here: freezing); everything else must be bug fixes validated independently. Tempting ideas (Tversky α/β tuning, hard-negative mining, domain-adversarial training) go in the queue, not in this run. Boring discipline, faster science.

### Q25. Why validate with tiny tests instead of just launching?

Because every check here catches failures that only show up hours into a GPU run otherwise: a crashed dataloader at batch 1, a constraint violated at step 1, a guardrail that halts into a crash. One forward+backward pass took seconds and caught a real bug (`collect_frozen_bn_layers` called with wrong argument type) that **would have killed Run 2 at launch**. Small tests, big savings.
