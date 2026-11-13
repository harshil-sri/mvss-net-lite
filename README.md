<div align="center">
  <img src="./docs/papers_and_reports/banner.svg" alt="MVSS-Net Lite Banner" width="100%">

  <br>

  ![Status](https://img.shields.io/badge/status-active--development-58a6ff?style=flat-square)
  ![Model](https://img.shields.io/badge/backbone-ResNet--34-a371f7?style=flat-square)
  ![Task](https://img.shields.io/badge/task-Forgery_Detection-3fb950?style=flat-square)
  ![Feature](https://img.shields.io/badge/feature-Self_Attention-8b949e?style=flat-square)

</div>

<br>

## Overview

Most forgery detectors give you a number: forged, or not, with some confidence score attached. They don't tell you why. **MVSS-Net Lite** is a document forgery detection system, built during a research internship at CDAC, that finds tampered regions at the pixel level and explains what it found instead of just handing over a score.

## What it does

- Detects manipulated regions in scanned documents (splicing, altered fields, tampered signatures) using a fusion-based deep learning model
- Explains each detection in plain language, backed by measurable evidence (edge inconsistencies, region confidence) instead of a single opaque score
- Lets reviewers ask questions about past results in natural language, instead of digging through logs
- Includes an evaluation harness that checks the model, the generated reports, and the retrieval system for accuracy over time

## How it works

```mermaid
flowchart LR
    A[Scanned Document] --> B[Detection Model]
    B --> C[Manipulated Regions + Evidence]
    C --> D[Forensic Report Generator]
    C --> E[(Prediction Archive)]
    D --> F[Readable Report + Heatmap]
    E --> G[Chat-based Query Layer]
    G --> H[Reviewer]
```

A document goes in. The model scores it region by region for signs of tampering. What comes out is a report a non-technical reviewer can actually read, with every past result stored and searchable afterward.

## Under the hood

| Layer | Approach |
|---|---|
| Detection model | MVSS-Net architecture with a ResNet-34 backbone and attention-based feature fusion |
| Evidence scoring | Edge-consistency and per-region confidence, not just one output number |
| Report generation | Model outputs converted into natural-language forensic summaries |
| Query layer | Retrieval-augmented querying over an archive of past predictions |

## Why this approach

A probability score isn't enough when someone actually has to act on a forgery flag. In legal review, financial checks, or academic document verification, you need to know what specifically looked wrong. This project treats explainability as part of the core system, not something bolted on after the fact.

## Project status

**Current State (July 2026):**
- **Model Training:** Stage 1 and Stage 2 training of MVSS-Net Lite are complete. Stage 2 successfully fine-tuned the model for document tampering with a high `pos_weight` to handle extreme class imbalance.
- **Inference Pipeline:** The FastAPI backend is fully operational. We recently resolved critical inference bugs related to image preprocessing (PIL interpolation matching) and threshold tuning (strict `0.97` segmentation threshold) to combat background inflation.
- **Evaluation & Review:** We have prepared a tightly masked, fully verified dataset of pure Document Forgeries (RTM) and Authentic Documents (MIDV500/RTM) that successfully pass end-to-end evaluation with 0 false positives for the latest mentor update.
- **Next Steps:** Scaling up the natural-language query layer and finalizing the database schema to store and retrieve historical forensic reports.

<br>

## Repository layout

| Path | Contents |
|---|---|
| `src/model/` | Network (`network.py`, `backbone.py`, `constrained_conv.py`, `fusion.py`, `pathb_network.py`), training (`train.py`), freezing schedule, configs and [architecture notes](src/model/ARCHITECTURE.md) |
| `src/model/scripts/` | Manifest building, leakage and overlap checks, Stage 2 training runs, pre-run validation |
| `src/data_pipeline/` | Loaders and converters for CASIA v2, DEFACTO, MIDV-500, RTM and Tampered-IC13, plus synthetic forgery generation |
| `src/diagnostics/` | Experiments run after Stage 2: blur probes, high-res patch evaluation on RTM, base-rate and calibration baselines, detection-head capacity, multi-seed aggregation |
| `src/reports/` | JSON results and plots from those experiments (seeds 7, 11, 42, 123, 202) |
| `src/backend/` | FastAPI service: inference, report generation, chat over past results |
| `scripts/` | Evaluation, launch and overnight run scripts, training patches, report and figure generation |
| `docs/` | Decision log, research notes and paper drafts |

## Model weights

The Stage 2 checkpoint (`stage2_mvss_lite_ep25_fp16.pt`, FP16, 25 epochs) is attached to the [v0.2-stage2 release](https://github.com/harshil-sri/mvss-net-lite/releases/tag/v0.2-stage2) rather than stored in the repository.

## Team

Built during a research internship at CDAC by Harshil Srivastav (model, training, evaluation), Mrigank Rana (data pipeline) and Rohit (backend).
