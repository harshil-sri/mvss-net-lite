import os
import sys
import torch
from torch.utils.data import DataLoader

from pipeline import ForgeryDataset, get_splits
from datasets.midv500_dataset import MIDV500Dataset, MIDV500_READY, MIDV500_ROOT

DOCTAMPER_IMAGES = 'data/doctamper/images'
DOCTAMPER_MASKS  = 'data/doctamper/masks'

CASIA_IMAGES = '/mnt/e/datasets/CasiaV2/CASIA2/Tp'
CASIA_MASKS  = '/mnt/e/datasets/CasiaV2/CASIA2/CASIA 2 Groundtruth'

BATCH_SIZE   = 4
IMAGE_SIZE   = (512, 512)


def sep(title):
    print(f'\n{"="*55}\n  {title}\n{"="*55}')


def check_batch(rgb, ela, mask, tag):
    assert rgb.shape  == torch.Size([BATCH_SIZE, 3, *IMAGE_SIZE]), f'[{tag}] bad RGB  shape: {rgb.shape}'
    assert ela.shape  == torch.Size([BATCH_SIZE, 3, *IMAGE_SIZE]), f'[{tag}] bad ELA  shape: {ela.shape}'
    assert mask.shape == torch.Size([BATCH_SIZE, 1, *IMAGE_SIZE]), f'[{tag}] bad Mask shape: {mask.shape}'
    unique = mask.unique().tolist()
    assert set(unique).issubset({0.0, 1.0}),                       f'[{tag}] mask not binary: {unique}'
    print(f'[{tag}] RGB  : {tuple(rgb.shape)}')
    print(f'[{tag}] ELA  : {tuple(ela.shape)}')
    print(f'[{tag}] Mask : {tuple(mask.shape)}  unique={unique}')


def test_doctamper():
    sep('TEST 1 — DocTamper')
    if not os.path.isdir(DOCTAMPER_IMAGES) or not os.listdir(DOCTAMPER_IMAGES):
        print('[DocTamper] SKIPPED — run generate_dummy_data.py first')
        return True
    dataset = ForgeryDataset(DOCTAMPER_IMAGES, DOCTAMPER_MASKS, IMAGE_SIZE)
    if len(dataset) < BATCH_SIZE:
        print(f'[DocTamper] SKIPPED — only {len(dataset)} samples')
        return True
    rgb, ela, mask = next(iter(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)))
    check_batch(rgb, ela, mask, 'DocTamper')
    train, val, test = get_splits(dataset)
    print(f'[DocTamper] Split  Train={len(train)}  Val={len(val)}  Test={len(test)}')
    print('[DocTamper] PASSED')
    return True


def test_casia():
    sep('TEST 2 — CASIA v2')
    if not os.path.isdir(CASIA_IMAGES) or not os.listdir(CASIA_IMAGES):
        print('[CASIA v2] SKIPPED — data not found at ' + CASIA_IMAGES)
        return True
    dataset = ForgeryDataset(CASIA_IMAGES, CASIA_MASKS, IMAGE_SIZE, max_samples=20)
    if len(dataset) < BATCH_SIZE:
        print(f'[CASIA v2] SKIPPED — only {len(dataset)} valid pairs found')
        return True
    rgb, ela, mask = next(iter(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)))
    check_batch(rgb, ela, mask, 'CASIA v2')
    print('[CASIA v2] PASSED')
    return True


def test_midv500():
    sep('TEST 3 — MIDV-500')
    if not MIDV500_READY:
        print('[MIDV-500] SKIPPED — MIDV500_READY=False in datasets/midv500_dataset.py')
        return True
    dataset = MIDV500Dataset(root_dir=MIDV500_ROOT, image_size=IMAGE_SIZE, forge_prob=0.5, max_samples=20)
    if len(dataset) < BATCH_SIZE:
        print(f'[MIDV-500] SKIPPED — only {len(dataset)} frames found')
        return True
    rgb, ela, mask = next(iter(DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)))
    check_batch(rgb, ela, mask, 'MIDV-500')
    train, val, test = get_splits(dataset)
    print(f'[MIDV-500] Split  Train={len(train)}  Val={len(val)}  Test={len(test)}')
    print('[MIDV-500] PASSED')
    return True


if __name__ == '__main__':
    print(f'\nDocForgery Pipeline Test Suite')
    print(f'PyTorch {torch.__version__}  |  Python {sys.version.split()[0]}')

    results = {
        'DocTamper': test_doctamper(),
        'CASIA v2':  test_casia(),
        'MIDV-500':  test_midv500(),
    }

    sep('SUMMARY')
    all_ok = True
    for name, ok in results.items():
        print(f'  [{"PASSED" if ok else "FAILED"}]  {name}')
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print('All tests passed. Pipeline ready.')
    else:
        print('Some tests failed.')
        sys.exit(1)
