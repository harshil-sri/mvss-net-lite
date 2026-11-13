"""
Convert Tampered-IC13 (bounding-box GT) into the repo's (image, mask) pair format.

Tampered-IC13 GT format: one .txt per image, each line "x1,y1,x2,y2,label"
where label==2 => tampered text, label==1 => real (untampered) text.
We build a BINARY tamper mask: pixels inside label==2 boxes = 255, else 0.
An image is "forged" iff it has >=1 label==2 box (authentic otherwise).

NOTE: these masks are COARSE (axis-aligned box fills), unlike the pixel-precise
masks in RTM/CASIA. This is inherent to the dataset and should be stated in the paper.

Output: <output>/<split>_<stem>.jpg  +  <output>/<split>_<stem>_mask.png
Also writes a standalone manifest (train/test as per the dataset's own split) so the
core reports/manifest.json used by completed experiments stays untouched.
"""
import os
import cv2
import json
import argparse
import numpy as np
from pathlib import Path


def parse_gt(gt_file: Path):
    """Return list of (x1, y1, x2, y2, label) ints. Tolerates blank lines."""
    boxes = []
    if not gt_file.exists():
        return boxes
    for raw in gt_file.read_text(errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 5:
            continue
        try:
            x1, y1, x2, y2 = (int(round(float(p))) for p in parts[:4])
            label = int(float(parts[4]))
        except ValueError:
            continue
        boxes.append((x1, y1, x2, y2, label))
    return boxes


def build_mask(h, w, boxes):
    mask = np.zeros((h, w), dtype=np.uint8)
    tampered = False
    for x1, y1, x2, y2, label in boxes:
        if label != 2:
            continue
        tampered = True
        xa, xb = sorted((x1, x2))
        ya, yb = sorted((y1, y2))
        xa = max(0, min(w, xa)); xb = max(0, min(w, xb))
        ya = max(0, min(h, ya)); yb = max(0, min(h, yb))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 255
    return mask, tampered


def convert(root: Path, out_dir: Path, repo_root: Path, rel_base: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = {"train": [], "test": []}
    counts = {"train": {"forged": 0, "auth": 0}, "test": {"forged": 0, "auth": 0}}

    for split in ("train", "test"):
        img_dir = root / f"{split}_img"
        gt_dir = root / f"{split}_gt"
        if not img_dir.is_dir():
            print(f"WARN: {img_dir} missing, skipping split {split}")
            continue
        img_files = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and not p.name.startswith(".")
        )
        for img_path in img_files:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"WARN: unreadable {img_path}")
                continue
            h, w = img.shape[:2]
            stem = img_path.stem
            boxes = parse_gt(gt_dir / f"{stem}.txt")
            mask, tampered = build_mask(h, w, boxes)

            out_stem = f"{split}_{stem}"
            out_img = out_dir / f"{out_stem}.jpg"
            out_mask = out_dir / f"{out_stem}_mask.png"
            cv2.imwrite(str(out_img), img)
            cv2.imwrite(str(out_mask), mask)

            entries[split].append({
                "image": f"{rel_base}/{out_stem}.jpg",
                "mask": f"{rel_base}/{out_stem}_mask.png",
                "dataset": "TamperedIC13",
                "forged": bool(tampered),
            })
            counts[split]["forged" if tampered else "auth"] += 1

    return entries, counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Format Tampered-IC13 into (image, mask) pairs")
    ap.add_argument("--root", required=True,
                    help="Path to the extracted 'Tampered-IC13' dir (contains train_img/test_img/train_gt/test_gt)")
    ap.add_argument("--output", default="data_pipeline/raw/TamperedIC13",
                    help="Output dir (repo-relative or absolute)")
    ap.add_argument("--manifest", default="reports/manifest_tampered_ic13.json",
                    help="Standalone manifest output (repo-relative)")
    ap.add_argument("--repo-root", default="/home/harshil/mvss-net-lite")
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    rel_base = args.output.strip("/") if not Path(args.output).is_absolute() else "data_pipeline/raw/TamperedIC13"
    out_dir = (Path(args.output) if Path(args.output).is_absolute() else repo_root / args.output)
    entries, counts = convert(Path(args.root).resolve(), out_dir, repo_root, rel_base)

    manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else repo_root / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(entries, indent=2))

    print("Tampered-IC13 conversion done.")
    for split in ("train", "test"):
        print(f"  {split}: {len(entries[split])} pairs "
              f"(forged={counts[split]['forged']}, auth={counts[split]['auth']})")
    print(f"  output dir : {out_dir}")
    print(f"  manifest   : {manifest_path}")
