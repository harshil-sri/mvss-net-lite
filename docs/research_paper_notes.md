# Research Paper Notes — MVSS-Net Lite / DocForge

Paper-ready material for the project. Everything under **"Verified results"**
is traceable to a file/test in this repo; anything hypothetical or pending is
flagged in the **Evidence ledger** (§8). Do not promote ledger items into
claims without running the experiment.

---

## 1. Contribution candidates

1. **A lightweight dual-branch forensic architecture for scanned-document
   forgery localization**, fusing RGB/structural and constrained-noise
   representations via CBAM attention at four scales, on a reduced ResNet-34
   backbone with dual output heads (region segmentation + tamper-edge map).
2. **Constrained-noise + appearance fusion at low compute** — Bayar-constrained
   convolution front-end (center = −1, surround Σ = +1) re-projected after every
   optimizer step, preserving a hard noise-residual inductive bias inside an
   otherwise trainable network.
3. **Explainability-by-construction**: pixel masks + edge evidence feed a
   natural-language report layer; every detection is auditable by a non-ML
   reviewer.
4. **An empirical study of shortcut learning under domain shift in document
   forensics**: quantified domain-identity shortcut (RTM-authentic FP rates)
   arising from naive domain-specific fine-tuning, plus two mitigation
   experiments (composition-balanced sampling; partial layer freezing) with a
   strict manifest-based evaluation protocol.
5. **A leakage-proof two-stage training protocol** for small-data fine-tuning:
   immutable manifest splits, live class-imbalance re-estimation, realized
   sampler-composition logging, checkpoint selection by validation-loss bottom
   cross-checked against domain FP probes.

---

## 2. Method summary (for §3 of the paper)

**Input**: scanned document crops, 256×256.

**Backbone (dual ResNet-34)**:
- *Noise branch*: Bayar ConstrainedConv2d (3→3, k=5) → ResNet-34 stem → stages 1–4.
- *Appearance/edge branch*: RGB → separate ResNet-34 stem → stages 1–4.

**Fusion**: per-stage CBAM (channel+spatial attention computed separately per branch;
fusion = concat + 1×1 conv) at channels {64,128,256,512}.

**Decoder**: progressive upsampling with skip concatenation (512→256→128→64), then
two heads: segmentation (1-ch logits) and edge (1-ch logits).

**Losses** (each head): BCE-with-logits(pos_weight, measured live) +
soft Tversky/Dice. Edge head uses Tversky α=0.3, β=0.7 (FN-weighted).

**Parameter budget (measured)**: total 45,717,619; frozen (Run 2 config)
2,696,033; trainable 43,021,586 (optimizer tensors: 200); frozen BN layers: 32.

**Two-stage curriculum**: Stage 1 on general forgery corpora (CASIAv2 + DEFACTO,
30,770 images of the unified pool); Stage 2 domain fine-tune on RTM + MIDV500
(24,050 images). Stage 2 Run 2 initializes from Stage 1 epoch 45 via weight
injection (`--init-weights`), never resume.

---

## 3. Verified results tables

### 3.1 Stage 1 checkpoint selection probe (threshold 0.9; source: `reports/stage1_checkpoint_selection.md`)

| Epoch | CASIAv2-auth FP | CASIAv2-forged TP | DEFACTO TP | Val loss | Train loss |
|---|---|---|---|---|---|
| 30 | 19/50 | 46/50 | 48/50 | — | — |
| 35 | 32/50 | 47/50 | 49/50 | 3.631 | 2.647 |
| 40 | 25/50 | 45/50 | 50/50 | 3.419 | 2.562 |
| **45** | **18/50** | **46/50** | **50/50** | 3.491 | 2.504 |
| 50 | 24/50 | 49/50 | 50/50 | 3.589 | 2.424 |

Selection rule: minimize FP subject to near-peak TP; corroborated by val-loss divergence at ep50. Selected: **ep45**.

### 3.2 Stage 2 Run 1 — overfitting trajectory (source: `reports/stage2_raw_data_report.md`, `reports/stage2_history.csv`)

| Epoch | Train total | Val total |
|---|---|---|
| 1 | 4.177 | 3.187 |
| 5 | 3.628 | **3.108 (bottom)** |
| 10 | 3.490 | 3.162 |
| 20 | 3.274 | 3.333 |
| 30 | 3.106 | 3.560 |
| 40 | 2.935 | 3.939 |
| 50 | 2.813 | **4.214** |

Val loss rises monotonically-ish for ~40 epochs while train falls; final gap 1.40. Launch provenance: resumed from an earlier ep10 checkpoint (not Stage 1 ep45); used runtime `random_split` (no saved indices ⇒ split unrecoverable); sampler composition never logged. Run discarded in full.

### 3.3 Domain-shortcut evidence (final Run 1 checkpoint)

| Probe set | FP @0.9 | FP @0.5 |
|---|---|---|
| RTM-authentic scans | **65.0%** | **76.5%** |
| MIDV500-authentic IDs | 2.0% | 7.0% |

Interpretation: model keys on "looks like a scanned paperwork" as proxy for "forged". Caveat for the paper: probe partition vs Run 1's unrecoverable train split ⇒ these are a likely-optimistic floor.

### 3.4 Label-resize ablation (synthetic 800px 1px-line, 1000→256 downscale; `check_mask_resize` output)

| Resize method | Preserved hline cols (/206) | Total mask px |
|---|---|---|
| NEAREST (old) | **1** (~0.5%) | 204 |
| BILINEAR>0 (fattening) | 206 | 1022 |
| Adaptive max-pool (fix) | 206 | 806 |
| Live dataset loader (= fix) | 206 | 806 |

NEAREST erases thin structures almost entirely; bilinear+threshold inflates width; max-pool preserves extent exactly. Prior incident: NEAREST silently destroyed ~87% of thin-edge labels until caught by this probe.

### 3.5 Sampler composition study (**reconstruction, not a Run 1 measurement**)

Run 1's realized draws were never logged (`reports/stage2_raw_data_report.md` §4 states this cannot be verified). Everything below is therefore a **reconstruction**: we read Run 1's weighting code and replayed both schemes against *today's* manifest train pool under a fresh sampler simulation. It tells us what the hardcoded weights *would* produce on this population — it does NOT tell us what Run 1 actually drew (different partition via `random_split`, unknown realized composition).

Measured population (manifest train split): rtm_forged **4,815** (25.13%), rtm_authentic **2,360** (12.32%), midv500 **11,986** (62.55%).

| Scheme | rtm_forged | rtm_authentic | midv500 | Δ from target |
|---|---|---|---|---|
| Target | 33.33% | 33.33% | 33.33% | — |
| Dynamic (measured counts) | 33.28% | 33.39% | 33.33% | ≤0.06pp |
| Legacy hardcoded (6000/3000/15050) — reconstructed | 33.92% | 32.81% | 33.28% | ≤0.58pp |

Finding (scoped): the legacy weights were misparameterized relative to the true population, but their ratios happen to land near-balanced on today's pool. This *weakens but does not falsify* the "sampler misconfiguration caused Run 1's shortcut" hypothesis — Run 1's actual draws remain unknowable. Balanced sampling alone should not be credited or blamed for shortcut outcomes without realized-composition logs.

### 3.6 Sampler exposure/repetition analysis (verified; `check_sampler` repetition block)

Per-epoch unique coverage and repeat exposure under the dynamic scheme (batch 8):

| Class | Pool | Unique drawn / epoch | Mean repeats per drawn image / epoch | Max repeats / epoch | Never drawn / epoch |
|---|---|---|---|---|---|
| rtm_forged | 4,815 | ~3,508 (72.8%) | ~1.79 | 6–8 | ~27% |
| rtm_authentic | 2,360 | ~2,205 (93.4%) | **~2.93** | 9–12 | ~6.6% |
| midv500 | 11,986 | ~4,954 (41.3%) | ~1.30 | 5–7 | ~58.7% |

Cumulative over 5 epochs: every rtm_authentic image is drawn at least once (mean **13.7 exposures/image**, max 28); MIDV500 coverage reaches 11,167/11,986 (93.2%). The authentic-RTM class is seen ~5× more often per image than MIDV500 per epoch — a quantified oversampling-repetition gradient that coexists with the domain shortcut and is a plausible secondary contributor to memorization of RTM-authentic appearance. *(Untested as an overfitting cause — flagged for a future single-variable experiment, e.g., per-class epoch-length decoupling.)*

### 3.7 Run 2 pre-launch validation (all PASS; `model/scripts/run2_validation/`)

- Manifest integrity: 54,805 entries; zero image/mask overlap across all split pairs; zero intra-split duplicates; zero missing files; Stage 2 subset 19,161/2,472/2,417 (train/val/test).
- Active-path AST scan: 7 reachable files (`model/train.py`, `model/network.py`, `model/backbone.py`, `model/fusion.py`, `model/constrained_conv.py`, `model/freezing.py`, `data_pipeline/dataset_loader.py`), 0 `random_split` occurrences.
- Wiring, freeze ON: stage1_ep45 loads clean (0 missing/unexpected keys); one fwd+bwd step → finite losses; 0 frozen params received grads; 200/200 trainable tensors did; Bayar constraint deviation ≤5.34e−05 post-step.
- Wiring, freeze OFF (plain pipeline): 297/297 trainable tensors received nonzero grads; losses finite; constraint intact (≤3.81e−05).
- End-to-end smoke ×2 variants (freeze ON and OFF), 4 optimizer steps each: live pos_weight rescan fires at startup; composition logged from batch 1 to persistent file in both; disk-guard save path exercised. Caveat: under `--smoke-test` the pos_weight scan covers only 3 batches, so scanned values vary between smoke runs (Seg 242.49 vs 75.08); a real launch scans the full epoch and is not affected.

---

## 4. Evaluation methodology notes (for §4)

- Domain-generalization probes: fixed image sets (e.g., 200/class where available) of casia_auth / casia_forged / defacto_forged / rtm_auth / midv_auth; report FP/TP at thresholds 0.5 AND 0.9 (operating-point sensitivity).
- Probes must be drawn from manifest val/test only, with zero-overlap stated numerically.
- Checkpoint selection: argmin(val total loss), never final epoch; confirm with FP probes.
- Report raw numbers side-by-side between runs; no aggregate verdicts.

---

## 5. Discussion angles

1. **Shortcut learning as the central failure mode of narrow-domain fine-tuning** — with dataset identity ~predictive of labels, even near-balanced *sampling ratios* may not prevent it (reconstructed sampler study §3.5) and oversampling repetition concentrates exposure on the minority authentic class (§3.6). Both are hypotheses until Run 2 probes test them.
2. **Freezing as implicit regularization** — cutting trainable capacity 45.7M→43.0M overall but more importantly pinning generic early features; hypothesis: reduces memorization channel for template identity. *(Pending Run 2 results.)*
3. **Label-resampling semantics matter at label scale** — max-pooling vs interpolation for binary masks is not cosmetic; it decides whether thin structures exist in training signal at all (87% label loss).
4. **Constraint re-projection cadence** — enforcing architectural constraints after every optimizer step keeps the noise extractor within its designed function class throughout training.
5. **Reproducibility infrastructure as a scientific control** — immutable manifests + realized-composition logging converted an unexplainable failure (Run 1) into measurable variables.

---

## 6. Related-work anchors

- MVSS-Net (multi-view multi-scale forgery segmentation) — base architecture lineage.
- Bayar & Stamm constrained convolution (CNN forensics; noise-residual inductive bias).
- CBAM (Woo et al.) — channel/spatial attention; our per-branch variant + concat fusion.
- ResNet-34 (He et al.) — Lite backbone choice; parameter budget rationale.
- Tversky loss (Salehi et al.) — asymmetric region overlap; FN weighting for forensic recall.
- Class imbalance: BCE pos_weight / focal-loss literature.
- Shortcut learning / spurious correlations (Geirhos et al.; model-vs-dataset-bias work).
- Transfer learning & freezing; BatchNorm train/eval statistics pitfalls (Ioffe & Szegedy).
- Document forensics datasets: CASIAv2, DEFACTO, RTM, MIDV500.

---

## 7. Suggested skeleton

1. Introduction — explainable document forensics; why localization beats scores.
2. Related work — forgery detection, constrained CNNs, attention fusion, imbalance handling.
3. Method — architecture (Fig: dual-backbone + CBAM fusion + decoder/heads), constraints, losses.
4. Two-stage training protocol — manifest splits, balanced sampling w/ realized logging, freezing design.
5. Experiments — Stage 1 selection (§3.1); Stage 2 Run 1 failure case study (§3.2–3.3) framed as shortcut-learning analysis; ablations: label resize (§3.4), sampler (§3.5); Run 2 freezing results *(pending)* vs Run 1 baselines.
6. System — explanation/report layer + retrieval QA (brief; differentiator for applied venue).
7. Limitations & future work — Tversky α/β sweep, hard-negative mining, domain-adversarial training, cross-dataset augmentation mixing; single-GPU scale limits; WSL2 dataloading constraint.
8. Reproducibility statement — validation suite inventory (`model/scripts/run2_validation/`), artifact list.

---

## 8. Evidence ledger (verified vs pending)

**Verified in-repo**: everything in §3 (each table cites its source file/script); architecture facts (§2) verifiable in code; freeze parameter budget; sampler population counts and draw statistics; manifest integrity numbers; exposure/repetition statistics (§3.6).

**Reconstruction (clearly labeled, not a measurement of Run 1)**: the legacy-scheme composition replay in §3.5 — Run 1's realized draws were never logged and its split is unrecoverable.

**Hypothesis / pending (do NOT cite as result)**:
- That layer freezing reduces RTM-authentic FP rate relative to Run 1 (awaits Run 2 + probe).
- That oversampling repetition of rtm_authentic (~2.9 exposures/image/epoch vs ~1.3 for MIDV500) contributes to memorization/shortcut behavior (quantified §3.6, causal role untested).
- Whether pos_weight values differ materially between partitions (222.60/929.22 under Run 1's accidental partition vs full-scan values at Run 2 launch — treat as sensitivity observation, not comparison).
- Any claim about MIDV500 masks being uniformly empty beyond the measured counts above (we verified class counts, not a per-image audit of every mask's emptiness semantics).
