# Stage 2 Run 4.1: MIDV500 Synthetic Forgery Injection

## Executive Summary
Exactly 5,000 synthetic MIDV500-forged images were generated using operations aligned with MIDV500's existing field bounding boxes (copy-move and field splicing), applied exclusively to a subset of the authentic MIDV500 images. The target composition sampler logic was updated, and a new unified manifest (`manifest_v2.json`) was generated with an 80/10/10 split ensuring zero image-level leakage.

---

## Generation Implementation
- **Source Method**: RTM's real forging methodology was not defined algorithmically in the repository. The synthetic generation was built using field annotations to create two tamper operations:
    1. **Field Copy-Move**: Clones a random field in the document template space, projects it to the image, and pastes it over another random field in the same image (with realistic blurring and a random brightness offset).
    2. **Field Splicing**: Projects a random field from a different image of the same document type (donor) into the target image's document structure.
- **Volume**: 5,000 images sampled uniformly from the `MIDV500` split of the old manifest were converted to forgeries. The remaining MIDV500 images were left authentic. 

---

## 6-Point Validation Suite Results

### 1. Manifest Integrity
```
Total entries: 59,805
Overlap Train/Val: 0
Overlap Train/Test: 0
Overlap Val/Test: 0
```
*Note: Evaluated across the combined pool `54,805 + 5,000`. Overlaps are strictly zero.*

### 2. Mask Pipeline Check
```
Successfully loaded synthetic batch via existing pipeline.
Mask shape: torch.Size([4, 1, 256, 256])
Max value: 1.0
```

### 3. Shortcut Audit
Low-level statistical signatures (Bayar noise-branch variance, laplacian high-frequency energy) were compared between real RTM tampered regions and the newly generated MIDV500 synthetic tampered regions.

```
RTM Forged (Real):
  Bayar Noise Var: 0.00609 ± 0.00586
  Bayar Noise Abs Mean: 0.04621
  Laplacian HF Energy: 0.47349

MIDV500 Forged (Synthetic):
  Bayar Noise Var: 0.00133 ± 0.00108
  Bayar Noise Abs Mean: 0.02728
  Laplacian HF Energy: 0.29559

FLAG: Large distributional gap detected! Synthetic noise variance is 78.2% different from real.
```

### 4. Visual QA Grid
The generated mask regions are distinct, and the document layouts are respected perfectly due to leveraging ground-truth MIDV-500 bounding box quads and homographies.
![QA Grid](/home/harshil/.gemini/antigravity-cli/brain/c5e2e37d-2ff8-4c4a-b29e-552f13f1c0d6/midv500_synthetic_qa_grid.jpg)

### 5. Sampler Empirical Ratio
The `TARGET_CLASSES` tuple was successfully extended to a 4-way split (`rtm_forged`, `rtm_auth`, `midv500_forged`, `midv500_auth`). Over 3,200 simulated batch draws via the `WeightedRandomSampler`, the empirical distribution was:
```
Empirical Ratio over 3200 draws:
  rtm_forged: 24.84%
  rtm_auth: 25.13%
  midv500_forged: 24.97%
  midv500_auth: 25.06%
```

### 6. Wiring Check
Ran a dummy train epoch utilizing the exact training dataloader settings and a live instantiation of `MVSSNetLite` with `torch.optim.Adam`.
```
Wiring check passed. Pipeline did not crash.
```

---
**Status**: The dataset is fully injected, `dataset_loader.py` is patched, and `manifest.json` points to the updated version (`manifest_v2.json`). It is structurally ready for training when you are.
