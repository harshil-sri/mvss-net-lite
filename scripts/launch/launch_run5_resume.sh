#!/usr/bin/env bash
# Resume stage2run5 from the best checkpoint (ep9_bestval, val 3.005).
# Restarts at epoch 10. Appends to existing history CSV.
set -e
cd /home/harshil/mvss-net-lite
tmux new-session -d -s stage2run5 \
  '/home/harshil/.venvs/global/bin/python3 -u -m model.train \
   --datasets RTM MIDV500 MIDV500_synthetic \
   --epochs 80 \
   --stage-name stage2run5 \
   --init-weights model/checkpoints/stage2run5_mvss_lite_ep9_bestval.pt \
   --resume \
   --use-balanced-sampler \
   --lr-schedule cosine \
   --early-stopping-patience 20 \
   --domain-adversarial \
   2>&1 | tee -a reports/stage2run5_train.log'
echo "Launched tmux session stage2run5. Tail with:"
echo "  tr '\r' '\n' < reports/stage2run5_train.log | tail -f"