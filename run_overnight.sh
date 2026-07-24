#!/bin/bash
set -e

# echo "=== STAGE 0: FORMATTING DATASETS ==="
# echo "Formatting CASIAv2..."
# python data_pipeline/format_casiav2.py --images data/CasiaV2/CASIA2 --masks "data/CasiaV2/CASIA2/CASIA 2 Groundtruth"
# 
# echo "Formatting DEFACTO..."
# python data_pipeline/format_defacto.py --images data/defacto/copymove_img/img --masks data/defacto/copymove_annotations/probe_mask
# 
# echo "Formatting RTM..."
# python data_pipeline/format_rtm.py --images data/RealTextManipulation/RealTextManipulation/JPEGImages --masks data/RealTextManipulation/RealTextManipulation/SegmentationClass
# 
# echo "Formatting MIDV-500..."
# python data_pipeline/format_midv500.py --input data/MIDV-500/midv500

echo "=== STAGE 1: TRAINING ON CASIAv2 & DEFACTO ==="
python -m model.train --datasets CASIAv2 DEFACTO --epochs 25 --stage-name stage1 --lr 1e-4

echo "=== STAGE 2: FINE-TUNING ON RTM & MIDV500 ==="
python -m model.train --datasets RTM MIDV500 --epochs 25 --stage-name stage2 --init-weights model/checkpoints/stage1_mvss_lite_ep25.pt --lr 1e-5

echo "=== ALL DONE! GOOD MORNING! ==="
