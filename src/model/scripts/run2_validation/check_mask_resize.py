"""
Check 4: mask-resize regression probe (synthetic 1px-line test) through the
LIVE data_pipeline/dataset_loader.py ForgeryDataset.

History: a NEAREST-interpolation resize once silently erased ~87% of thin edge
labels; the fix is F.adaptive_max_pool2d. A repo reorg once reverted it
silently - so this probe runs against the live loader no matter what.

Method: build a synthetic 1000x1000 image + mask with (a) an 800-px-long 1-px
horizontal line and (b) a 1-px diagonal line; pass through the live dataset at
crop_size=256; measure preserved pixel mass and line extent. Compare against
NEAREST (old bug) and BILINEAR>0 (fattening bug) baselines computed on the same
synthetic mask, mirroring debug_memories/test_pooling.py.

Run from repo root:
    python -m model.scripts.run2_validation.check_mask_resize
"""
import os
import tempfile

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from data_pipeline.dataset_loader import ForgeryDataset

SIZE = 1000
CROP = 256
LINE_LEN = 800  # cols 100..900


def make_synthetic(tmp):
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[500, 100:900] = 255          # horizontal 1-px line, 800 px long
    for i in range(100, 900):         # diagonal 1-px line, ~800 px long
        mask[i, i] = 255
    img = np.full((SIZE, SIZE, 3), 245, dtype=np.uint8)
    # put some texture so the image looks non-degenerate
    cv2.circle(img, (300, 300), 80, (180, 180, 180), -1)
    img_path = os.path.join(tmp, 'probe.jpg')
    mask_path = os.path.join(tmp, 'probe_mask.png')
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask)
    return img_path, mask_path, mask


def baseline_stats(mask):
    mask_pil = Image.fromarray(mask)
    nearest = (np.array(mask_pil.resize((CROP, CROP), Image.NEAREST)) > 127).astype(np.uint8)
    bilinear = (np.array(mask_pil.resize((CROP, CROP), Image.BILINEAR)) > 0).astype(np.uint8)
    t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    maxpool = (F.adaptive_max_pool2d(t, (CROP, CROP)).squeeze().numpy() > 0).astype(np.uint8)
    return {'NEAREST(old bug)': nearest, 'BILINEAR>0(fattening)': bilinear,
            'MAXPOOL(reference fix)': maxpool}


def line_extent(binary_256):
    """How many columns of the horizontal line survived, in output pixels."""
    row = binary_256[128, :]  # 500/1000*256 = 128
    return int(row.sum())


def diag_extent(binary_256):
    """Pixels along the diagonal trace region that survived."""
    cnt = 0
    for out_y in range(CROP):
        src_x = int((out_y + 0.5) / CROP * SIZE)  # x==y in source
        if 100 <= src_x < 900:
            lo = max(0, src_x - 4)
            hi = min(SIZE, src_x + 5)
            if binary_256[out_y, int(lo / SIZE * CROP):int(hi / SIZE * CROP)].any():
                cnt += 1
    return cnt


def main():
    tmp = tempfile.mkdtemp(prefix='mask_probe_', dir='/tmp/opencode')
    img_path, mask_path, mask_src = make_synthetic(tmp)
    print(f"Synthetic assets in: {tmp}")
    print(f"Source mask {SIZE}x{SIZE}: positive px={(mask_src>0).sum()} "
          f"(horizontal line {LINE_LEN}px + diagonal ~{LINE_LEN}px)")

    print("\n=== Baselines computed directly on the synthetic mask ===")
    expected_maxpool_h = line_extent(baseline_stats(mask_src)['MAXPOOL(reference fix)'])
    for name, m in baseline_stats(mask_src).items():
        print(f"{name:24s}: total_px={int(m.sum()):4d} | hline_cols={line_extent(m):3d}/{expected_maxpool_h} "
              f"| diag_px~{diag_extent(m):3d}")

    print("\n=== LIVE ForgeryDataset (is_train=False, crop_size=256) ===")
    ds = ForgeryDataset([(img_path, mask_path)], crop_size=CROP, is_train=False)
    img_t, mask_t, edge_t = ds[0]
    mask_np = mask_t.squeeze().numpy()
    mask_bin = (mask_np > 0.5).astype(np.uint8)
    live_total = int(mask_bin.sum())
    live_h = line_extent(mask_bin)
    live_diag = diag_extent(mask_bin)
    print(f"live mask tensor: shape={tuple(mask_t.shape)} total_px={live_total} "
          f"hline_cols={live_h} diag_px~{live_diag}")
    print(f"live edge tensor: shape={tuple(edge_t.shape)} positive_px={int(edge_t.sum())}")

    print("\n=== LIVE ForgeryDataset (is_train=True, deterministic seed) ===")
    torch.manual_seed(0); np.random.seed(0)
    import random as _r; _r.seed(0)
    ds_tr = ForgeryDataset([(img_path, mask_path)], crop_size=CROP, is_train=True)
    kept = []
    for trial in range(20):
        _, m_t, e_t = ds_tr[0]
        mb = (m_t.squeeze().numpy() > 0.5).astype(np.uint8)
        kept.append(int(mb.sum()))
    print(f"20 train-mode samples: preserved_px min={min(kept)} max={max(kept)} "
          f"mean={sum(kept)/len(kept):.1f} (aug rotations/crops may reduce; must stay >> 0)")

    ok = live_h >= int(expected_maxpool_h * 0.95) and live_total > 0 and min(kept) > 50
    print(f"\nExpected hline preservation ≈ full 800px extent scaled to "
          f"{CROP}px => {expected_maxpool_h} cols (max-pool keeps every column the line touches)")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == '__main__':
    main()
