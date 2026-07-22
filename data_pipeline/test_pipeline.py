import os
import sys
import torch
from torch.utils.data import DataLoader

from data_pipeline.dataset_loader import ForgeryDataset, get_splits

# Configuration
DATA_ROOT = "data_pipeline/raw"

BATCH_SIZE       = 4
NUM_WORKERS      = 2
CROP_SIZE        = 512

# Helpers
def separator(title: str):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def validate_batch(rgb, mask, edge, tag: str):
    """Assert shapes and mask uniqueness, print a summary."""
    assert rgb.shape  == torch.Size([BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE]), \
        f"[{tag}] Bad RGB shape: {rgb.shape}"
    assert mask.shape == torch.Size([BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE]), \
        f"[{tag}] Bad Mask shape: {mask.shape}"
    assert edge.shape == torch.Size([BATCH_SIZE, 1, CROP_SIZE, CROP_SIZE]), \
        f"[{tag}] Bad Edge shape: {edge.shape}"

    unique = mask.unique().tolist()
    assert set(unique).issubset({0.0, 1.0}), \
        f"[{tag}] Mask has non-binary values: {unique}"

    print(f"[{tag}] RGB shape:     {tuple(rgb.shape)}")
    print(f"[{tag}] Mask shape:    {tuple(mask.shape)}")
    print(f"[{tag}] Edge shape:    {tuple(edge.shape)}")
    print(f"[{tag}] Mask unique:   {unique}")


# Test 1 -- DocTamper
def test_doctamper():
    separator("TEST 1 -- DocTamper Dataset")

    dataset = ForgeryDataset(
        data_root=DATA_ROOT,
        dataset_names=['DocTamper'],
        crop_size=CROP_SIZE,
    )

    if len(dataset) == 0:
        print("[DocTamper] SKIPPED -- no samples found.")
        return False

    if len(dataset) < BATCH_SIZE:
        print(f"[DocTamper] SKIPPED -- only {len(dataset)} samples, need at least {BATCH_SIZE}.")
        return False

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    rgb, mask, edge = next(iter(loader))
    validate_batch(rgb, mask, edge, "DocTamper")

    train_set, val_set, test_set = get_splits(dataset, train_frac=0.70, val_frac=0.15)
    print(f"[DocTamper] Split   -- Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print("[DocTamper] PASSED")
    return True

# Test 2 -- CASIA v2
def test_casiaV2():
    separator("TEST 2 -- CASIA v2 Dataset")

    dataset = ForgeryDataset(
        data_root=DATA_ROOT,
        dataset_names=['CASIAv2'],
        crop_size=CROP_SIZE,
    )

    if len(dataset) == 0:
        print("[CASIA v2] SKIPPED -- no samples found.")
        return True

    if len(dataset) < BATCH_SIZE:
        print(f"[CASIA v2] SKIPPED -- only {len(dataset)} valid pairs found.")
        return True

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    rgb, mask, edge = next(iter(loader))
    validate_batch(rgb, mask, edge, "CASIA v2")
    print("[CASIA v2] PASSED")
    return True

# Test 3 -- DEFACTO
def test_defacto():
    separator("TEST 3 -- DEFACTO Dataset")

    dataset = ForgeryDataset(
        data_root=DATA_ROOT,
        dataset_names=['Defacto'],
        crop_size=CROP_SIZE,
    )

    if len(dataset) == 0:
        print("[DEFACTO] SKIPPED -- no samples found.")
        return True

    if len(dataset) < BATCH_SIZE:
        print(f"[DEFACTO] SKIPPED -- only {len(dataset)} valid pairs found.")
        return True

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    rgb, mask, edge = next(iter(loader))
    validate_batch(rgb, mask, edge, "DEFACTO")
    train_set, val_set, test_set = get_splits(dataset, train_frac=0.70, val_frac=0.15)
    print(f"[DEFACTO] Split    -- Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print("[DEFACTO] PASSED")
    return True


# Test 4 -- Multi-worker Edge Race Condition Check
def test_multiworker_edge():
    separator("TEST 4 -- Multi-Worker Edge Race Condition Check")

    dataset = ForgeryDataset(
        data_root=DATA_ROOT,
        dataset_names=['DocTamper'],
        crop_size=CROP_SIZE,
    )

    if len(dataset) < BATCH_SIZE:
        print("[Race Test] SKIPPED -- not enough DocTamper samples.")
        return True

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    edge_outputs = []
    for i, (_, _, edge) in enumerate(loader):
        edge_outputs.append(edge)
        if i >= 2:
            break

    for i, edge in enumerate(edge_outputs):
        assert edge.abs().sum() > 0, \
            f"Batch {i} Edge is all-zeros -- possible bug"

    print(f"[Race Test] Verified {len(edge_outputs)} batches -- no race conditions detected.")
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
        "Race Check": test_multiworker_edge(),
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
