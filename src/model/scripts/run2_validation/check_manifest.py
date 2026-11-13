"""
Check 1: Manifest integrity for Stage 2 Run 2.

- Loads reports/manifest.json
- Confirms zero overlap between train/val/test image-path sets
- Confirms no intra-split duplicates
- Confirms counts match expected totals (54,805 overall)
- Reports RTM+MIDV500 subset counts used for Stage 2
- Verifies every referenced file exists on disk

Run from repo root:
    python -m model.scripts.run2_validation.check_manifest
"""
import json
import os
from collections import Counter

EXPECTED_TOTAL = 54805


def main():
    with open('reports/manifest.json', 'r') as f:
        manifest = json.load(f)

    splits = {}
    print("=== Per-split raw counts ===")
    for split in ('train', 'val', 'test'):
        entries = manifest[split]
        paths = [s['image'] for s in entries]
        sets_ = set(paths)
        dup = len(paths) - len(sets_)
        per_ds = Counter(s['dataset'] for s in entries)
        splits[split] = sets_
        print(f"{split}: entries={len(entries)} unique_images={len(sets_)} intra_split_dupes={dup}")
        print(f"  per-dataset: {dict(per_ds)}")

    grand = sum(len(manifest[s]) for s in ('train', 'val', 'test'))
    print(f"\nGrand total entries: {grand} (expected {EXPECTED_TOTAL}) -> "
          f"{'MATCH' if grand == EXPECTED_TOTAL else 'MISMATCH'}")

    print("\n=== Pairwise overlap check (image paths) ===")
    tv = splits['train'] & splits['val']
    tt = splits['train'] & splits['test']
    vt = splits['val'] & splits['test']
    print(f"train & val  overlap: {len(tv)}")
    print(f"train & test overlap: {len(tt)}")
    print(f"val  & test overlap: {len(vt)}")

    # Mask-path overlap too (a leaked mask is as bad as a leaked image)
    msets = {s: set(x['mask'] for x in manifest[s]) for s in ('train', 'val', 'test')}
    mtv = msets['train'] & msets['val']
    mtt = msets['train'] & msets['test']
    mvt = msets['val'] & msets['test']
    print(f"train & val  mask overlap: {len(mtv)}")
    print(f"train & test mask overlap: {len(mtt)}")
    print(f"val  & test mask overlap: {len(mvt)}")

    print("\n=== Stage 2 subset (RTM + MIDV500 only) ===")
    for split in ('train', 'val', 'test'):
        sub = [s for s in manifest[split] if s['dataset'] in ('RTM', 'MIDV500')]
        per_ds = Counter(s['dataset'] for s in sub)
        print(f"{split}: total={len(sub)} | {dict(per_ds)}")

    sub_train = {s['image'] for s in manifest['train'] if s['dataset'] in ('RTM', 'MIDV500')}
    sub_val = {s['image'] for s in manifest['val'] if s['dataset'] in ('RTM', 'MIDV500')}
    sub_test = {s['image'] for s in manifest['test'] if s['dataset'] in ('RTM', 'MIDV500')}
    print(f"subset overlaps: train&val={len(sub_train & sub_val)} "
          f"train&test={len(sub_train & sub_test)} val&test={len(sub_val & sub_test)}")

    print("\n=== File existence check ===")
    missing_img, missing_mask = [], []
    for split in ('train', 'val', 'test'):
        for s in manifest[split]:
            if not os.path.exists(s['image']):
                missing_img.append((split, s['image']))
            if not os.path.exists(s['mask']):
                missing_mask.append((split, s['mask']))
    print(f"missing images: {len(missing_img)}")
    for row in missing_img[:10]:
        print("  ", row)
    print(f"missing masks:  {len(missing_mask)}")
    for row in missing_mask[:10]:
        print("  ", row)

    ok = (grand == EXPECTED_TOTAL and not tv and not tt and not vt
          and not mtv and not mtt and not mvt
          and not missing_img and not missing_mask)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == '__main__':
    main()
