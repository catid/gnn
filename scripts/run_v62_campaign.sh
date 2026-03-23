#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

ISSUE_ID="${1:-gnn-5j7}"

count_complete() {
  local schedule="$1"
  local expected_steps="$2"
  python3 - "$schedule" "$expected_steps" <<'PY'
from pathlib import Path
import json, re, sys

schedule = sys.argv[1]
expected_steps = int(sys.argv[2])
pat = re.compile(r"v62-(?P<regime>[^-]+)-(?P<arm>[a-z0-9_]+)-32-(?P<schedule>p|m|l|xl)(?:-(?P<tag>[^-]+))?-s(?P<seed>\d+)$")
count = 0
for run in Path("runs").glob("*-v62-*"):
    if not run.is_dir():
        continue
    m = pat.search(run.name)
    if not m or m.group("schedule") != schedule:
        continue
    metrics = run / "metrics.jsonl"
    config = run / "config.yaml"
    last = run / "last.pt"
    if not metrics.exists() or not config.exists() or not last.exists():
        continue
    step = 0
    for line in metrics.read_text().splitlines():
        if line.strip():
            step = max(step, int(json.loads(line).get("step", 0)))
    if step >= expected_steps:
        count += 1
print(count)
PY
}

wait_for_count() {
  local schedule="$1"
  local expected_count="$2"
  local expected_steps="$3"
  while true; do
    local count
    count="$(count_complete "$schedule" "$expected_steps")"
    echo "v62 watcher: schedule=$schedule complete=$count/$expected_count"
    if [[ "$count" -ge "$expected_count" ]]; then
      break
    fi
    sleep 30
  done
}

extract_lr() {
  local arm="$1"
  python3 - "$arm" <<'PY'
import json, sys
from pathlib import Path

arm = sys.argv[1]
obj = json.loads(Path("reports/summary_metrics_v62.json").read_text())
print(obj["selected_lrs"][arm])
PY
}

extract_ambiguity() {
  python3 - <<'PY'
import json
from pathlib import Path

obj = json.loads(Path("reports/summary_metrics_v62.json").read_text())
print("1" if obj["ambiguity"]["triggered"] else "0")
for regime in obj["ambiguity"]["candidate_regimes"][:2]:
    print(regime)
PY
}

run_arm_phase() {
  local gpu_id="$1"
  local arm="$2"
  local schedule="$3"
  local lr_scale="$4"
  shift 4
  local regimes=("$1")
  shift 1
  local seeds=("$1")
  shift 1
  # shellcheck disable=SC2206
  local regime_list=(${regimes[0]})
  # shellcheck disable=SC2206
  local seed_list=(${seeds[0]})
  for regime in "${regime_list[@]}"; do
    for seed in "${seed_list[@]}"; do
      export EXTRA_ARGS="--lr-scale ${lr_scale}"
      export TRAIN_LOG_TO_FILE=1
      export GPU_ID="${gpu_id}"
      export SEED="${seed}"
      export TRAIN_LOG_PATH="runs/v62-${regime}-${arm}-32-${schedule}-s${seed}.console.log"
      bash "scripts/train_v62_${regime}_${arm}_32_${schedule}.sh"
    done
  done
}

wait_for_count p 24 420
python scripts/run_v62_eval_sweep.py --regimes core t1 t2a hmix --schedules p --arms visitonly visit_taskgrad_half --batches 64 --best-only
python scripts/build_v62_report.py

V_LR="$(extract_lr visitonly)"
VT_LR="$(extract_lr visit_taskgrad_half)"
echo "v62 watcher: selected LRs V=${V_LR} VT-0.5=${VT_LR}"

(
  run_arm_phase 0 visitonly m "${V_LR}" "core t1 t1r t2a t2b t2c hmid hmix" "1234 2234 3234"
) &
PID_V_M=$!
(
  run_arm_phase 1 visit_taskgrad_half m "${VT_LR}" "core t1 t1r t2a t2b t2c hmid hmix" "1234 2234 3234"
) &
PID_VT_M=$!
wait "$PID_V_M" "$PID_VT_M"

python scripts/run_v62_eval_sweep.py --regimes core t1 t1r t2a t2b t2c hmid hmix --schedules m --arms visitonly visit_taskgrad_half --batches 96
python scripts/build_v62_report.py

(
  run_arm_phase 0 visitonly l "${V_LR}" "core t1 t2a hmix" "4234 5234"
) &
PID_V_L=$!
(
  run_arm_phase 1 visit_taskgrad_half l "${VT_LR}" "core t1 t2a hmix" "4234 5234"
) &
PID_VT_L=$!
wait "$PID_V_L" "$PID_VT_L"

python scripts/run_v62_eval_sweep.py --regimes core t1 t2a hmix --schedules l --arms visitonly visit_taskgrad_half --batches 128
python scripts/build_v62_report.py

mapfile -t ambiguity_info < <(extract_ambiguity)
AMBIG_TRIGGER="${ambiguity_info[0]}"
if [[ "${AMBIG_TRIGGER}" == "1" && "${#ambiguity_info[@]}" -ge 3 ]]; then
  REGIME_A="${ambiguity_info[1]}"
  REGIME_B="${ambiguity_info[2]}"
  echo "v62 watcher: ambiguity breaker on ${REGIME_A} ${REGIME_B}"
  (
    run_arm_phase 0 visitonly xl "${V_LR}" "${REGIME_A} ${REGIME_B}" "6234 7234"
  ) &
  PID_V_XL=$!
  (
    run_arm_phase 1 visit_taskgrad_half xl "${VT_LR}" "${REGIME_A} ${REGIME_B}" "6234 7234"
  ) &
  PID_VT_XL=$!
  wait "$PID_V_XL" "$PID_VT_XL"

  python scripts/run_v62_eval_sweep.py --regimes "${REGIME_A}" "${REGIME_B}" --schedules xl --arms visitonly visit_taskgrad_half --batches 128
  python scripts/build_v62_report.py
fi

pytest -q
bd update "${ISSUE_ID}" --append-notes "Completed v62 static selector tie-break campaign with pilots, 8-regime M matrix, L anchors, optional XL, eval sweeps, and final report."
bd close "${ISSUE_ID}"
