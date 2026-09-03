# Diagnostic Report: Blur Setting Probe

## Blur Setting A: Current Default (Gaussian Blur 5x5, sigma=1.5)

| Group | Bayar Var (Global) | Bayar Var (Delta) | Bayar Mean (Global) | Bayar Mean (Delta) | Lap HF (Global) | Lap HF (Delta) |
|---|---|---|---|---|---|---|
| Real RTM-Forged | 0.00604 | 0.00272 (±0.00534) | 0.04583 | 0.02180 (±0.03174) | 0.46936 | 0.21311 (±0.31875) |
| MIDV500-Forged (Synthetic) | 0.00560 | 0.00382 (±0.00468) | 0.05265 | 0.03206 (±0.02666) | 0.51907 | 0.31964 (±0.24888) |
| MIDV500-Authentic Baseline | 0.00556 | 0.00378 (±0.00458) | 0.05020 | 0.02964 (±0.02370) | 0.47090 | 0.27175 (±0.20902) |

**Delta Quartile Breakdowns (Forged - Authentic)**
* **Real RTM-Forged**:
  * Bayar Var: `[-0.00039, 0.00199, 0.00554]`
  * Bayar Mean: `[-0.00013, 0.02289, 0.04152]`
  * Lap HF: `[0.00567, 0.21862, 0.42692]`
* **Synthetic MIDV500-Forged**:
  * Bayar Var: `[0.00020, 0.00288, 0.00637]`
  * Bayar Mean: `[0.01088, 0.03137, 0.05067]`
  * Lap HF: `[0.12487, 0.30738, 0.47698]`
* **MIDV500-Authentic**:
  * Bayar Var: `[0.00043, 0.00268, 0.00603]`
  * Bayar Mean: `[0.01241, 0.02845, 0.04271]`
  * Lap HF: `[0.12043, 0.26652, 0.38840]`


## Blur Setting B: No Blur (Binary exact mask)

| Group | Bayar Var (Global) | Bayar Var (Delta) | Bayar Mean (Global) | Bayar Mean (Delta) | Lap HF (Global) | Lap HF (Delta) |
|---|---|---|---|---|---|---|
| Real RTM-Forged | 0.00604 | 0.00272 (±0.00534) | 0.04583 | 0.02180 (±0.03174) | 0.46936 | 0.21311 (±0.31875) |
| MIDV500-Forged (Synthetic) | 0.00596 | 0.00419 (±0.00498) | 0.05469 | 0.03410 (±0.02768) | 0.53871 | 0.33923 (±0.25673) |
| MIDV500-Authentic Baseline | 0.00556 | 0.00378 (±0.00458) | 0.05020 | 0.02964 (±0.02370) | 0.47090 | 0.27175 (±0.20902) |

**Delta Quartile Breakdowns (Forged - Authentic)**
* **Real RTM-Forged**:
  * Bayar Var: `[-0.00039, 0.00199, 0.00554]`
  * Bayar Mean: `[-0.00013, 0.02289, 0.04152]`
  * Lap HF: `[0.00567, 0.21862, 0.42692]`
* **Synthetic MIDV500-Forged**:
  * Bayar Var: `[0.00044, 0.00311, 0.00687]`
  * Bayar Mean: `[0.01308, 0.03316, 0.05196]`
  * Lap HF: `[0.15066, 0.32024, 0.50582]`
* **MIDV500-Authentic**:
  * Bayar Var: `[0.00043, 0.00268, 0.00603]`
  * Bayar Mean: `[0.01241, 0.02845, 0.04271]`
  * Lap HF: `[0.12043, 0.26652, 0.38840]`


## Blur Setting C: Intermediate (Gaussian Blur 3x3, sigma=0.75)

| Group | Bayar Var (Global) | Bayar Var (Delta) | Bayar Mean (Global) | Bayar Mean (Delta) | Lap HF (Global) | Lap HF (Delta) |
|---|---|---|---|---|---|---|
| Real RTM-Forged | 0.00604 | 0.00272 (±0.00534) | 0.04583 | 0.02180 (±0.03174) | 0.46936 | 0.21311 (±0.31875) |
| MIDV500-Forged (Synthetic) | 0.00577 | 0.00399 (±0.00480) | 0.05367 | 0.03308 (±0.02714) | 0.52932 | 0.32986 (±0.25306) |
| MIDV500-Authentic Baseline | 0.00556 | 0.00378 (±0.00458) | 0.05020 | 0.02964 (±0.02370) | 0.47090 | 0.27175 (±0.20902) |

**Delta Quartile Breakdowns (Forged - Authentic)**
* **Real RTM-Forged**:
  * Bayar Var: `[-0.00039, 0.00199, 0.00554]`
  * Bayar Mean: `[-0.00013, 0.02289, 0.04152]`
  * Lap HF: `[0.00567, 0.21862, 0.42692]`
* **Synthetic MIDV500-Forged**:
  * Bayar Var: `[0.00025, 0.00302, 0.00670]`
  * Bayar Mean: `[0.01171, 0.03255, 0.05161]`
  * Lap HF: `[0.13294, 0.31654, 0.49615]`
* **MIDV500-Authentic**:
  * Bayar Var: `[0.00043, 0.00268, 0.00603]`
  * Bayar Mean: `[0.01241, 0.02845, 0.04271]`
  * Lap HF: `[0.12043, 0.26652, 0.38840]`
