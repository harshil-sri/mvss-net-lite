#!/usr/bin/env bash
# Round 2 diagnostic pipeline
set -u
cd /home/harshil/mvss-net-lite/src
export PYTHONPATH=/home/harshil/mvss-net-lite/src
PY=/home/harshil/.venvs/global/bin/python3
mkdir -p reports/logs
FAILS=0

echo "=== calibration_baseline seed 42 start $(date) ==="
if CB_SEED=42 $PY diagnostics/calibration_baseline.py > reports/logs/calibration_baseline_seed42.log 2>&1; then
  echo "=== calibration_baseline seed 42 OK $(date) ==="
else
  echo "=== calibration_baseline seed 42 FAILED rc=$? $(date) ==="
  FAILS=$((FAILS+1))
fi

for MODE in det erm random; do
  echo "=== dethead_capacity_ablation mode $MODE start $(date) ==="
  if ABL_MODE=$MODE ABL_SEED=42 $PY diagnostics/dethead_capacity_ablation.py > reports/logs/dethead_capacity_${MODE}_seed42.log 2>&1; then
    echo "=== dethead_capacity_ablation mode $MODE OK $(date) ==="
  else
    echo "=== dethead_capacity_ablation mode $MODE FAILED rc=$? $(date) ==="
    FAILS=$((FAILS+1))
  fi
done

echo "=== gated_pixel_eval start $(date) ==="
if GP_SEED=42 $PY diagnostics/gated_pixel_eval.py > reports/logs/gated_pixel_seed42.log 2>&1; then
  echo "=== gated_pixel_eval OK $(date) ==="
else
  echo "=== gated_pixel_eval FAILED rc=$? $(date) ==="
  FAILS=$((FAILS+1))
fi

echo "=== shuffled_probe_control start $(date) ==="
if SP_SEED=42 $PY diagnostics/shuffled_probe_control.py > reports/logs/shuffled_probe_seed42.log 2>&1; then
  echo "=== shuffled_probe_control OK $(date) ==="
else
  echo "=== shuffled_probe_control FAILED rc=$? $(date) ==="
  FAILS=$((FAILS+1))
fi

for S in 11 202; do
  echo "=== baserate_experiment seed $S start $(date) ==="
  if BR_SEED=$S $PY diagnostics/baserate_experiment.py > reports/logs/baserate_seed${S}.log 2>&1; then
    echo "=== baserate_experiment seed $S OK $(date) ==="
  else
    echo "=== baserate_experiment seed $S FAILED rc=$? $(date) ==="
    FAILS=$((FAILS+1))
  fi
done

echo "ALL ROUND 2 RUNS DONE fails=$FAILS $(date)"
exit $FAILS
