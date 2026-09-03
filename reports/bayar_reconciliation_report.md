# Global Bayar-Variance Metric Reconciliation Probe

## Step 1 & 2: File Paths & Line Ranges
The exact original implementations of both methodologies have been reconstructed into a single script for direct comparison on the same images.
- **Original Audit Code (Implementation 1)**: `reconciliation_probe.py`, lines 16-37 (reconstructed from `validate_midv500_synthetic.py` via `dataset_loader.py`).
- **Blur-Probe Code (Implementation 2)**: `reconciliation_probe.py`, lines 39-55 (reconstructed from `probe_blur.py`).

## Step 3: Methodology Verbatim Quotes

### Implementation 1: Original Audit (Dataloader equivalent)
- **Region:** Forged-region-only via mask (`mask > 0`).
- **Channels:** Converted to Grayscale for the Bayar convolution.
- **Preprocessing/Normalization:** Images were loaded with PIL and resized using `Image.BILINEAR`. Masks were pooled using `F.adaptive_max_pool2d`. Image arrays were transposed and divided by `255.0`.

**Verbatim Code:**
```python
# Loading and Preprocessing
img_pil_resized = img_pil.resize((256, 256), Image.BILINEAR)
mask_t = torch.from_numpy(np.array(mask_pil)).float().unsqueeze(0).unsqueeze(0)
mask_pil_resized = Image.fromarray(F.adaptive_max_pool2d(mask_t, (256, 256)).squeeze().numpy().astype(np.uint8))
img_tensor_orig = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0

# Variance Calculation
gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
noise = F.conv2d(gray, bayar_kernel, padding=1)
mask = mask_tensor > 0
tampered_pixels = noise[mask]
return tampered_pixels.var().item()
```

### Implementation 2: Blur-Probe Report (Manual cv2 equivalent)
- **Region:** Forged-region-only via mask (`mask > 0`).
- **Channels:** Converted to Grayscale for the Bayar convolution.
- **Preprocessing/Normalization:** Images were loaded with OpenCV, resized using `cv2.resize` (default `INTER_LINEAR`), and converted to tensor using `torchvision.transforms.ToTensor()`. Masks were resized using `cv2.resize` with `INTER_NEAREST`.

**Verbatim Code:**
```python
# Loading and Preprocessing
t_img = cv2.resize(t_img, (256, 256))
m_img = cv2.resize(m_img, (256, 256), interpolation=cv2.INTER_NEAREST)
t_img_t = to_tensor(cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))

# Variance Calculation
gray = 0.299 * img_tensor[:, 0:1] + 0.587 * img_tensor[:, 1:2] + 0.114 * img_tensor[:, 2:3]
noise = F.conv2d(gray, bayar_kernel, padding=1)
m1 = mask > 0
v1 = noise[m1].var().item()
```

## Step 4: Fixed Sample
A fixed seed of 100 images (50 synthetic MIDV500-forged + 50 real RTM-forged) was sampled. The IDs/paths have been saved to `bayar_reconciliation_sample_manifest.json` in the workspace.

## Step 5: Dual Methodology Computation
*Computed over the fixed 100-image sample (50 MIDV500-Synthetic, 50 RTM-Forged).*

### Implementation 1: Original Audit (PIL Resize / Pool Mask)
| Region | Source | Mean | Std | Min | Q1 | Median | Q3 | Max |
|--------|--------|------|-----|-----|----|--------|----|-----|
| Masked | MIDV500 | 0.00138 | 0.00107 | 0.00010 | 0.00060 | 0.00101 | 0.00185 | 0.00416 |
| Masked | RTM | 0.00624 | 0.00522 | 0.00000 | 0.00195 | 0.00584 | 0.00838 | 0.02193 |
| Whole | MIDV500 | 0.00063 | 0.00019 | 0.00027 | 0.00051 | 0.00063 | 0.00076 | 0.00122 |
| Whole | RTM | 0.00341 | 0.00212 | 0.00086 | 0.00178 | 0.00301 | 0.00437 | 0.01176 |

### Implementation 2: Blur-Probe (CV2 Resize / CV2 Mask)
| Region | Source | Mean | Std | Min | Q1 | Median | Q3 | Max |
|--------|--------|------|-----|-----|----|--------|----|-----|
| Masked | MIDV500 | 0.00586 | 0.00557 | 0.00015 | 0.00166 | 0.00373 | 0.00860 | 0.02456 |
| Masked | RTM | 0.03101 | 0.02672 | 0.00000 | 0.01042 | 0.02830 | 0.04294 | 0.13375 |
| Whole | MIDV500 | 0.00169 | 0.00076 | 0.00051 | 0.00117 | 0.00159 | 0.00200 | 0.00388 |
| Whole | RTM | 0.01263 | 0.00990 | 0.00202 | 0.00590 | 0.00873 | 0.01686 | 0.05251 |

## Step 6: Match Check
Does the original audit's reported 0.00133 (Synthetic) / 0.00609 (RTM) figures match either the whole-image or forged-region-only numbers from Step 5, within 5% tolerance?

- **MIDV500-Synthetic (Original: 0.00133)**
  - Matches Implementation 1 (Masked Region: 0.00138)? **YES** (3.7% difference)
- **Real RTM-Forged (Original: 0.00609)**
  - Matches Implementation 1 (Masked Region: 0.00624)? **YES** (2.4% difference)
