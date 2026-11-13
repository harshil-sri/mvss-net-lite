#!/bin/bash
tmux new-session -d -s stage2run4 'cd ~/mvss-net-lite && /home/harshil/.venvs/global/bin/python3 -u -m model.train \
  --datasets RTM MIDV500 \
  --epochs 80 \
  --stage-name stage2run4 \
  --init-weights model/checkpoints/stage1_mvss_lite_ep45.pt \
  --use-balanced-sampler \
  --lr-schedule cosine \
  --early-stopping-patience 20 \
  --domain-adversarial \
  2>&1 | tee reports/stage2run4_train.log'
