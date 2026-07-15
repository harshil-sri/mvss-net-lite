# MVSS-Net Lite — Document Forgery Detection

A PyTorch implementation of a document forgery detection pipeline built on the MVSS-Net architecture. The model takes a document image, generates an Error Level Analysis (ELA) map, and produces a pixel-level binary segmentation mask identifying forged regions.

---

## Overview

Document forgery detection is the task of identifying manipulated regions in scanned or photographed documents — such as altered text, copy-moved fields, or spliced content. This project implements a two-stream approach:

- **Stream 1 (RGB):** The raw document image, normalized with ImageNet statistics.
- **Stream 2 (ELA):** An Error Level Analysis map that reveals JPEG compression inconsistencies introduced by forgery operations.

Both streams are fed into the MVSS-Net backbone, which outputs an edge map and a pixel-level segmentation mask.

---

## Project Structure

```
DocForgery/
├── pipeline.py                     # Core dataset class and ELA generation
├── preprocessing.py                # Augmentation transforms (synced across RGB, ELA, mask)
├── test_pipeline.py                # Validation suite for all datasets
├── generate_dummy_data.py          # Simple synthetic document generator
├── generate_realistic_forgeries.py # Realistic forgery generator (copy-move, splicing)
├── models/
│   └── mvssnet.py                  # MVSS-Net architecture
├── datasets/
│   └── midv500_dataset.py          # MIDV-500 adapter with synthetic forgery injection
└── data/
    ├── doctamper/
    │   ├── images/
    │   └── masks/
    ├── casiaV2/
    │   ├── images/
    │   └── masks/
    └── midv500/
```

---

## Datasets

| Dataset | Type | Notes |
|---|---|---|
| DocTamper (synthetic) | Generated locally | Run `generate_dummy_data.py` or `generate_realistic_forgeries.py` |
| CASIA v2 | Real-world forgeries | Images in `.tif`, masks with `_gt.png` suffix |
| MIDV-500 | Identity documents | Authentic frames; forgeries synthesized at load time from quad annotations |

---

## Setup

**Install dependencies**
```bash
pip install torch torchvision pillow numpy opencv-python
```

**Generate synthetic DocTamper data**
```bash
python generate_dummy_data.py
# or for more realistic data:
python generate_realistic_forgeries.py
```

**Validate the pipeline**
```bash
python test_pipeline.py
```

Expected output:
```
[DocTamper] RGB shape:   (4, 3, 512, 512)
[DocTamper] ELA shape:   (4, 3, 512, 512)
[DocTamper] Mask shape:  (4, 1, 512, 512)
[DocTamper] Mask unique: [0.0, 1.0]
[DocTamper] PASSED
```

---

## Pipeline

### ELA Generation

Error Level Analysis re-compresses the image at a fixed JPEG quality and computes the pixel-wise difference between the original and re-compressed versions. Forged regions, which have a different compression history, show higher error levels.

```python
from pipeline import generate_ela
ela_image = generate_ela('path/to/image.jpg', quality=90)
```

### Dataset Class

```python
from pipeline import ForgeryDataset, get_splits

dataset = ForgeryDataset(
    image_dir='data/doctamper/images',
    mask_dir='data/doctamper/masks',
    image_size=(512, 512)
)

train_set, val_set, test_set = get_splits(dataset, train_frac=0.70, val_frac=0.15)
```

Each sample returns:
- `rgb_tensor` — shape `(3, 512, 512)`, ImageNet-normalized
- `ela_tensor` — shape `(3, 512, 512)`, raw difference values
- `mask_tensor` — shape `(1, 512, 512)`, binary `{0.0, 1.0}`

### Balanced Sampling

The pipeline computes per-sample weights based on whether each mask contains forged pixels, then uses `WeightedRandomSampler` to ensure equal representation of forged and authentic samples in every training batch.

---

## Model — MVSS-Net

The MVSS-Net architecture consists of:

- **ResNet-50 backbone** — extracts multi-scale feature maps at four levels
- **Sobel edge stream** — fixed edge-detection filters applied to each feature level to highlight boundary artefacts at forgery borders
- **BayarConv2d** — a constrained convolution that learns to suppress image content and focus on noise residuals, improving sensitivity to digital manipulation
- **Edge Refinement Blocks (ERB)** — combine skip connections across scales to produce a refined edge/boundary map
- **Dual Attention Head** — combines position attention (spatial self-attention) and channel attention before the final segmentation prediction

The model returns two outputs:
- `res1` — predicted edge map at the forgery boundaries
- `x0` — pixel-level forgery segmentation mask

---

## Preprocessing and Augmentation

`preprocessing.py` provides synchronized augmentation that applies the **same** spatial transformation (flip, rotation, crop) to the RGB image, ELA map, and mask simultaneously — ensuring the mask remains aligned after any geometric operation.

```python
from preprocessing import get_train_transforms, get_val_transforms

synced, rgb_tf, ela_tf, mask_tf = get_train_transforms()
rgb_aug, ela_aug, mask_aug = synced(rgb_pil, ela_pil, mask_pil)
```

---

## Acknowledgements

- MVSS-Net paper: *Multi-View Feature Learning for Forgery Detection* 
- MIDV-500 dataset: Bulatov et al.
- CASIA v2 dataset: Chinese Academy of Sciences
