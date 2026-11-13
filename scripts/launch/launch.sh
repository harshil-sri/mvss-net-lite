#!/bin/bash
tmux new-session -d -s stage2run3 'cd ~/mvss-net-lite && /home/harshil/.venvs/global/bin/python3 -u -m model.train \
  --datasets RTM MIDV500 \
  --epochs 50 \
  --stage-name stage2run3 \
  --init-weights model/checkpoints/stage1_mvss_lite_ep45.pt \
  --use-balanced-sampler \
  --lr-schedule cosine \
  --early-stopping-patience 8 \
  --domain-adversarial \
  2>&1 | tee reports/stage2run3_train.log'
