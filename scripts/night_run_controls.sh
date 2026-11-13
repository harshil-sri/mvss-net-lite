#!/usr/bin/env bash
# det-head balanced/unbalanced controls x 3 seeds, then calibration baseline
set -u
cd /home/harshil/mvss-net-lite/src
export PYTHONPATH=/home/harshil/mvss-net-lite/src
PY=/home/harshil/.venvs/global/bin/python3
mkdir -p reports/logs
FAILS=0
for S in 42 123 7; do
  for B in 1 0; do
    MODE=$([ "$B" = 1 ] && echo bal || echo unbal)
    echo "=== deth $MODE seed $S start $(date) ==="
    if DH_SEED=$S DH_BALANCED=$B $PY diagnostics/detection_head_experiment.py > reports/logs/deth_${MODE}_seed${S}.log 2>&1; then
      echo "=== deth $MODE seed $S OK $(date) ==="
    else
      echo "=== deth $MODE seed $S FAILED rc=$? $(date) ==="
      FAILS=$((FAILS+1))
    fi
  done
done
echo "=== calibration seed 42 start $(date) ==="
if CB_SEED=42 $PY diagnostics/calibration_baseline.py > reports/logs/calibration_seed42.log 2>&1; then
  echo "=== calibration seed 42 OK $(date) ==="
else
  echo "=== calibration seed 42 FAILED rc=$? $(date) ==="
  FAILS=$((FAILS+1))
fi
echo "ALL CONTROL RUNS DONE fails=$FAILS $(date)"
exit $FAILS
