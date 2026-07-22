#!/bin/bash
set -e

echo "=== STAGE 0: FORMATTING DATASETS ==="
echo "Formatting CASIAv2..."
python data_pipeline/format_casiav2.py --images data/CASIA2 --masks "data/CASIA2/CASIA 2 Groundtruth"

echo "Formatting DEFACTO..."
python data_pipeline/format_defacto.py

echo "Formatting RTM..."
python data_pipeline/format_rtm.py

echo "Formatting MIDV-500..."
python data_pipeline/format_midv500.py --input data/midv500

echo "=== STAGE 1: TRAINING ON CASIAv2 & DEFACTO ==="
python model/train.py --datasets CASIAv2 DEFACTO --epochs 25 --stage-name stage1

echo "=== STAGE 2: FINE-TUNING ON RTM & MIDV500 ==="
python model/train.py --datasets RTM MIDV500 --epochs 25 --stage-name stage2 --init-weights model/checkpoints/stage1_mvss_lite_ep25.pt

echo "=== ALL DONE! GOOD MORNING! ==="
