#!/usr/bin/env bash
# runs the base-rate sweep for 3 seeds sequentially; logs to reports/logs
set -u
cd /home/harshil/mvss-net-lite/src
export PYTHONPATH=/home/harshil/mvss-net-lite/src
PY=/home/harshil/.venvs/global/bin/python3
mkdir -p reports/logs
FAILS=0
for S in 42 123 7; do
  echo "=== baserate seed $S start $(date) ==="
  if BR_SEED=$S $PY diagnostics/baserate_experiment.py > reports/logs/baserate_seed${S}.log 2>&1; then
    echo "=== baserate seed $S OK $(date) ==="
  else
    echo "=== baserate seed $S FAILED rc=$? $(date) ==="
    FAILS=$((FAILS+1))
  fi
done
echo "ALL BASE-RATE SEEDS DONE fails=$FAILS $(date)"
exit $FAILS
