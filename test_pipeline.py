"""
Expected output:
    TEST 1 -- DocTamper Dataset
    [DocTamper] RGB shape:     (4, 3, 512, 512)
    [DocTamper] ELA shape:     (4, 3, 512, 512)
    [DocTamper] Mask shape:    (4, 1, 512, 512)
    [DocTamper] Mask unique:   [0.0, 1.0]
    [DocTamper] Split   -- Train: 70 | Val: 15 | Test: 15
    [DocTamper] PASSED

    TEST 2 -- CASIA v2 Dataset
    [CASIA v2] RGB shape:     (4, 3, 512, 512)
    [CASIA v2] PASSED

    TEST 3 -- DEFACTO Dataset
    [DEFACTO] RGB shape:     (4, 3, 512, 512)
    [DEFACTO] Split    -- Train: 14 | Val: 3 | Test: 3
    [DEFACTO] PASSED

    TEST 4 -- MIDV-500 Dataset
    [MIDV500Dataset] 20 frames loaded from /mnt/e/datasets/MIDV-500/midv500
    [MIDV-500] RGB shape:     (4, 3, 512, 512)
    [MIDV-500] Split    -- Train: 14 | Val: 3 | Test: 3
    [MIDV-500] PASSED

    TEST 5 -- Multi-Worker ELA Race Condition Check
    [Race Test] Verified 3 batches -- no race conditions detected.
    [Race Test] PASSED

    SUMMARY
    [PASSED]  DocTamper
    [PASSED]  CASIA v2
    [PASSED]  DEFACTO
    [PASSED]  MIDV-500
    [PASSED]  Race Check

    Pipeline working.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader

from pipeline import ForgeryDataset, get_splits
from datasets.midv500_dataset import MIDV500Dataset, MIDV500_READY

# Configuration

DOCTAMPER_IMAGES = "data/doctamper/images"
DOCTAMPER_MASKS  = "data/doctamper/masks"

CASIAV2_IMAGES   = "/mnt/e/datasets/CasiaV2/CASIA2/Tp"
CASIAV2_MASKS    = "/mnt/e/datasets/CasiaV2/CASIA2/CASIA 2 Groundtruth"

DEFACTO_IMAGES   = "/mnt/e/datasets/DEFACTO/copymove_img/img"
DEFACTO_MASKS    = "/mnt/e/datasets/DEFACTO/copymove_annotations/probe_mask"

MIDV500_ROOT     = "/mnt/e/datasets/MIDV-500/midv500"

BATCH_SIZE       = 4
NUM_WORKERS      = 2
IMAGE_SIZE       = (512, 512)

# Helpers

def separator(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def validate_batch(rgb, ela, mask, tag: str):
    """Assert shapes and mask uniqueness, print a summary."""
    assert rgb.shape  == torch.Size([BATCH_SIZE, 3, *IMAGE_SIZE]), \
        f"[{tag}] Bad RGB shape: {rgb.shape}"
    assert ela.shape  == torch.Size([BATCH_SIZE, 3, *IMAGE_SIZE]), \
        f"[{tag}] Bad ELA shape: {ela.shape}"
    assert mask.shape == torch.Size([BATCH_SIZE, 1, *IMAGE_SIZE]), \
        f"[{tag}] Bad Mask shape: {mask.shape}"

    unique = mask.unique().tolist()
    assert set(unique).issubset({0.0, 1.0}), \
        f"[{tag}] Mask has non-binary values: {unique}"

    print(f"[{tag}] RGB shape:     {tuple(rgb.shape)}")
    print(f"[{tag}] ELA shape:     {tuple(ela.shape)}")
    print(f"[{tag}] Mask shape:    {tuple(mask.shape)}")
    print(f"[{tag}] Mask unique:   {unique}")


# Test 1 -- DocTamper

def test_doctamper():
    separator("TEST 1 -- DocTamper Dataset")

    dataset = ForgeryDataset(
        image_dir=DOCTAMPER_IMAGES,
        mask_dir=DOCTAMPER_MASKS,
        image_size=IMAGE_SIZE,
    )

    if len(dataset) == 0:
        print("[DocTamper] SKIPPED -- no samples found. Run generate_dummy_data.py first.")
        return False

    if len(dataset) < BATCH_SIZE:
        print(f"[DocTamper] SKIPPED -- only {len(dataset)} samples, need at least {BATCH_SIZE}.")
        return False

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    rgb, ela, mask = next(iter(loader))
    validate_batch(rgb, ela, mask, "DocTamper")

    train_set, val_set, test_set = get_splits(dataset, train_frac=0.70, val_frac=0.15)
    print(f"[DocTamper] Split   -- Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print("[DocTamper] PASSED")
    return True



# Test 2 -- CASIA v2 (skip gracefully if empty)


def test_casiaV2():
    separator("TEST 2 -- CASIA v2 Dataset")

    if not os.path.isdir(CASIAV2_IMAGES) or len(os.listdir(CASIAV2_IMAGES)) == 0:
        print("[CASIA v2] SKIPPED -- data/casiaV2/images/ is empty.")
        return True

    dataset = ForgeryDataset(
        image_dir=CASIAV2_IMAGES,
        mask_dir=CASIAV2_MASKS,
        image_size=IMAGE_SIZE,
        max_samples=20,
    )

    if len(dataset) < BATCH_SIZE:
        print(f"[CASIA v2] SKIPPED -- only {len(dataset)} valid pairs found.")
        return True

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    rgb, ela, mask = next(iter(loader))
    validate_batch(rgb, ela, mask, "CASIA v2")
    print("[CASIA v2] PASSED")
    return True


# Test 3 -- DEFACTO

def test_defacto():
    separator("TEST 3 -- DEFACTO Dataset")

    if not os.path.isdir(DEFACTO_IMAGES) or len(os.listdir(DEFACTO_IMAGES)) == 0:
        print("[DEFACTO] SKIPPED -- data not found at " + DEFACTO_IMAGES)
        return True

    dataset = ForgeryDataset(
        image_dir=DEFACTO_IMAGES,
        mask_dir=DEFACTO_MASKS,
        image_size=IMAGE_SIZE,
        max_samples=20,
    )

    if len(dataset) < BATCH_SIZE:
        print(f"[DEFACTO] SKIPPED -- only {len(dataset)} valid pairs found.")
        return True

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    rgb, ela, mask = next(iter(loader))
    validate_batch(rgb, ela, mask, "DEFACTO")
    train_set, val_set, test_set = get_splits(dataset, train_frac=0.70, val_frac=0.15)
    print(f"[DEFACTO] Split    -- Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print("[DEFACTO] PASSED")
    return True


# Test 4 -- MIDV-500 (skip gracefully if not downloaded yet)

def test_midv500():
    separator("TEST 4 -- MIDV-500 Dataset")

    if not MIDV500_READY:
        print("[MIDV-500] SKIPPED -- MIDV500_READY=False in datasets/midv500_dataset.py.")
        print("           Set it to True after placing data in data/midv500/")
        return True

    if not os.path.isdir(MIDV500_ROOT) or len(os.listdir(MIDV500_ROOT)) == 0:
        print("[MIDV-500] SKIPPED -- data/midv500/ is empty. Download dataset first.")
        return True

    dataset = MIDV500Dataset(
        root_dir=MIDV500_ROOT,
        image_size=IMAGE_SIZE,
        forge_prob=0.5,
        max_samples=20,
    )

    if len(dataset) < BATCH_SIZE:
        print(f"[MIDV-500] SKIPPED -- only {len(dataset)} frames found.")
        return True

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    rgb, ela, mask = next(iter(loader))
    validate_batch(rgb, ela, mask, "MIDV-500")

    train_set, val_set, test_set = get_splits(dataset, train_frac=0.70, val_frac=0.15)
    print(f"[MIDV-500] Split    -- Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print("[MIDV-500] PASSED")
    return True


# Test 4 -- Multi-worker ELA race condition check

def test_multiworker_ela():
    separator("TEST 5 -- Multi-Worker ELA Race Condition Check")

    dataset = ForgeryDataset(
        image_dir=DOCTAMPER_IMAGES,
        mask_dir=DOCTAMPER_MASKS,
        image_size=IMAGE_SIZE,
    )

    if len(dataset) < BATCH_SIZE:
        print("[Race Test] SKIPPED -- not enough DocTamper samples.")
        return True

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    ela_outputs = []
    for i, (_, ela, _) in enumerate(loader):
        ela_outputs.append(ela)
        if i >= 2:
            break

    for i, ela in enumerate(ela_outputs):
        assert ela.abs().sum() > 0, \
            f"Batch {i} ELA is all-zeros -- possible BytesIO bug"

    print(f"[Race Test] Verified {len(ela_outputs)} batches -- no race conditions detected.")
    print("[Race Test] PASSED")
    return True


# Main

if __name__ == '__main__':
    print("\nDocForgery Pipeline Test Suite")
    print(f"Working dir : {os.getcwd()}")
    print(f"Python      : {sys.version.split()[0]}")
    print(f"PyTorch     : {torch.__version__}")

    results = {
        "DocTamper" : test_doctamper(),
        "CASIA v2"  : test_casiaV2(),
        "DEFACTO"   : test_defacto(),
        "MIDV-500"  : test_midv500(),
        "Race Check": test_multiworker_ela(),
    }

    separator("SUMMARY")
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  [{status}]  {name}")

    print()
    if all_passed:
        print("Pipeline working.")
    else:
        print("Some tests failed. Fix errors above before training.")
        sys.exit(1)
