#!/bin/bash
# Representation sweep: HANNA with ChemBERTa +/- RDKit descriptors.
# Runs jobs sequentially; one failure logs [fail] and continues to the next.
# Per-job log -> reports/logs/<suffix>.log ; one-line result -> reports/logs/sweep_status.log
set -u

PY=/opt/homebrew/Caskroom/miniforge/base/envs/train_hanna/bin/python
cd "$(dirname "$0")/.." || exit 1

N_EPOCHS=150
PATIENCE_EARLY=20
SCHED_PATIENCE=8
DEVICE=mps

STATUS=reports/logs/sweep_status.log
mkdir -p reports/logs
: > "$STATUS"

# job list: "<descriptor-set> <seed> <suffix>"
# with-theta (frozen) re-runs of the 3 configs not already available on this arch.
# (full -> reports/metrics/full_with_fine_tune_layer.json; full_only -> fullonly_s0.json,
#  both already on the with-theta arch with identical protocol.)
JOBS=(
  "none         0 none_s0"
  "curated      0 curated_s0"
  "curated_only 0 curonly_s0"
)

echo "[sweep-start] $(date '+%Y-%m-%d %H:%M:%S')  ${#JOBS[@]} jobs" | tee -a "$STATUS"

for job in "${JOBS[@]}"; do
  read -r dset seed suffix <<< "$job"
  log="reports/logs/${suffix}.log"
  if [ -f "reports/metrics/${suffix}.json" ]; then
    echo "[skip] ${suffix} (already has metrics json)" | tee -a "$STATUS"
    continue
  fi
  echo "[start] ${suffix} (set=${dset} seed=${seed}) $(date '+%H:%M:%S')" | tee -a "$STATUS"
  $PY -m experiments.train_with_descriptors \
      --descriptor-set "$dset" --suffix "$suffix" --seed "$seed" \
      --n_epochs "$N_EPOCHS" --patience_early "$PATIENCE_EARLY" \
      --patience "$SCHED_PATIENCE" --device "$DEVICE" > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[fail] ${suffix} rc=${rc} $(date '+%H:%M:%S') (see ${log})" | tee -a "$STATUS"
    continue
  fi
  # pull headline numbers from the JSON the python wrote
  summary=$($PY - "$suffix" <<'PYEOF'
import json, sys
s = sys.argv[1]
try:
    m = json.load(open(f"reports/metrics/{s}.json"))
    bv = (m.get("loss_history") or {}).get("best_epoch")
    print(f"emb={m['embedding_dim']} meanMAE={m['test_mean_system_MAE']:.4f} "
          f"medMAE={m['test_median_system_MAE']:.4f} overallMAE={m['test_overall_MAE']:.4f} "
          f"bestEp={bv} t={m['wall_time_sec']}s")
except Exception as e:
    print(f"(no json: {e})")
PYEOF
)
  echo "[done] ${suffix} ${summary} $(date '+%H:%M:%S')" | tee -a "$STATUS"
done

echo "[sweep-complete] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$STATUS"
