# Phase 1: Mask-Pipeline Provenance Check

## Step 1: RTM-forged mask provenance
- **File**: `data_pipeline/dataset_loader.py`
- **Last-Modified**: `2026-08-31 07:36:46`
- **Lines**: 70-71 (training) and identical 86-87 (validation/probe)
- **Exact Code**:
```python
            mask_t = torch.from_numpy(np.array(mask_pil)).float().unsqueeze(0).unsqueeze(0)
            mask_pil = Image.fromarray(F.adaptive_max_pool2d(mask_t, (self.crop_size, self.crop_size)).squeeze().numpy().astype(np.uint8))
```

## Step 2 & 3: MIDV500-synthetic-forged and MIDV500-authentic mask provenance
- **File**: Cannot be cited.
- **Reason**: The ad-hoc scripts used to generate the synthetic forgeries (`generate_midv500_synthetic.py`) and run the blur-probe (`probe_blur.py`) were permanently deleted from the disk in my cleanup step at the end of the previous session. I cannot cite live files, timestamps, or exact lines for them because they no longer exist in the workspace.

## Step 4 & 5: Divergence Check (Live Pipeline)
Because the generation and probe scripts were deleted, I cannot confirm a divergence based on them. However, looking strictly at the remaining **live training pipeline** (`dataset_loader.py`):

In the live pipeline, **they do not diverge in how they are resized.**

- All images (both RTM and MIDV500-synthetic) are read from disk by `dataset_loader.py`.
- Both groups have their masks processed by the exact same code blocks (lines 70-71 or 86-87), meaning both use `F.adaptive_max_pool2d` for mask resizing and `PIL.Image.BILINEAR` for image resizing.

The divergence observed in the reconciliation probe was purely an artifact of the ad-hoc `probe_blur.py` script (which I used to generate the blur tables). In that script, I evaluated RTM by passing it through the dataloader (`adaptive_max_pool2d`), but I evaluated MIDV500 by processing it inline in memory using `cv2.resize` with `INTER_NEAREST`. This measurement flaw was confined to the probe scripts and does not exist in the live training dataloader.

**Status:** Phase 1 does not confirm a divergence in the live training pipeline. Consequently, as per instructions, I have stopped here and will not proceed to Phase 2.
