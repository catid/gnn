#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

TASK_ID="${1:-}"
if [[ -z "${TASK_ID}" ]]; then
  echo "usage: $0 <beads-task-id>" >&2
  exit 1
fi

run_train() {
  local gpu="$1"
  local regime="$2"
  local condition="$3"
  local schedule="$4"
  local seed="$5"
  local lr_scale="$6"
  local tag="$7"
  local script="scripts/train_v67_collision_${regime}_${condition}_visit_taskgrad_half_d_32_${schedule}.sh"
  local run_name="v67-collision-${regime}-${condition}-visit_taskgrad_half_d-32-${schedule}"
  if [[ -n "${tag}" ]]; then
    run_name="${run_name}-${tag}"
  fi
  run_name="${run_name}-s${seed}"
  GPU_ID="${gpu}" SEED="${seed}" RUN_NAME="${run_name}" EXTRA_ARGS="--lr-scale ${lr_scale}" bash "${script}"
}

python scripts/gen_v67_configs.py
python -m compileall apsgnn apsgnn/probes.py scripts/gen_v67_configs.py scripts/run_v67_eval_sweep.py scripts/build_v67_report.py tests/test_v67_cache_summary_fusion.py
pytest -q tests/test_v67_cache_summary_fusion.py
pytest -q

# 100-step timing/profile run on the strongest failure regime from v66.
GPU_ID=0 SEED=1234 RUN_NAME="v67-collision-c2-baseline-visit_taskgrad_half_d-32-p-profile-s1234" TRAIN_STEPS_OVERRIDE=100 bash scripts/train_v67_collision_c2_baseline_visit_taskgrad_half_d_32_p.sh

# Fair-shot LR pilots on C2 cache-on.
for lr_tag in lr06 lr08 lr10; do
  case "${lr_tag}" in
    lr06) lr="0.6" ;;
    lr08) lr="0.8" ;;
    lr10) lr="1.0" ;;
  esac
  run_train 0 c2 baseline p 1234 "${lr}" "${lr_tag}" &
  pid0=$!
  run_train 1 c2 fusion p 1234 "${lr}" "${lr_tag}" &
  pid1=$!
  wait "${pid0}" "${pid1}"
done

python scripts/run_v67_eval_sweep.py --regimes c2 --conditions baseline fusion --schedules p --batches 48
python scripts/build_v67_report.py

BASELINE_LR="$(python - <<'PY'
import json, pathlib
obj=json.loads(pathlib.Path('reports/summary_metrics_v67.json').read_text())
print(obj['pilot_choices']['baseline'].get('lr_multiplier', 0.6))
PY
)"
FUSION_LR="$(python - <<'PY'
import json, pathlib
obj=json.loads(pathlib.Path('reports/summary_metrics_v67.json').read_text())
print(obj['pilot_choices']['fusion'].get('lr_multiplier', 0.6))
PY
)"

# Main matched collision validation.
for regime in c1 c2; do
  for seed in 2234 3234; do
    run_train 0 "${regime}" baseline m "${seed}" "${BASELINE_LR}" "" &
    pid0=$!
    run_train 1 "${regime}" fusion m "${seed}" "${FUSION_LR}" "" &
    pid1=$!
    wait "${pid0}" "${pid1}"
  done
done

python scripts/run_v67_eval_sweep.py --regimes c1 c2 --conditions baseline fusion --schedules m --batches 96
python scripts/run_v67_eval_sweep.py --regimes c2 --conditions baseline fusion --schedules m --batches 96 --collect-probes
python scripts/build_v67_report.py

STRONGEST_REGIME="$(python - <<'PY'
import json, pathlib
obj=json.loads(pathlib.Path('reports/summary_metrics_v67.json').read_text())
print(obj.get('strongest_regime', 'c2'))
PY
)"

# Fresh rerun on the strongest regime.
run_train 0 "${STRONGEST_REGIME}" baseline l 4234 "${BASELINE_LR}" "" &
pid0=$!
run_train 1 "${STRONGEST_REGIME}" fusion l 4234 "${FUSION_LR}" "" &
pid1=$!
wait "${pid0}" "${pid1}"

python scripts/run_v67_eval_sweep.py --regimes "${STRONGEST_REGIME}" --conditions baseline fusion --schedules l --batches 128
python scripts/build_v67_report.py
pytest -q tests/test_v67_cache_summary_fusion.py

bd update "${TASK_ID}" --append-notes "Completed v67 cache summary fusion rescue: added the query-at-home home-cache fusion path, ran fair-shot pilots plus matched C1/C2 validation and a fresh rerun, rebuilt v67 summary/report, and reran focused verification."
bd close "${TASK_ID}"
