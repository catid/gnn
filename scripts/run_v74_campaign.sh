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
  local script="scripts/train_v74_collision_${regime}_${condition}_visit_taskgrad_half_d_32_${schedule}.sh"
  local run_name="v74-collision-${regime}-${condition}-visit_taskgrad_half_d-32-${schedule}"
  if [[ -n "${tag}" ]]; then
    run_name="${run_name}-${tag}"
  fi
  run_name="${run_name}-s${seed}"
  GPU_ID="${gpu}" SEED="${seed}" RUN_NAME="${run_name}" EXTRA_ARGS="--lr-scale ${lr_scale}" bash "${script}"
}

python scripts/gen_v74_configs.py
python -m compileall apsgnn apsgnn/probes.py scripts/gen_v74_configs.py scripts/run_v74_eval_sweep.py scripts/build_v74_report.py tests/test_v74_max_augmented_output_readout.py
pytest -q tests/test_v74_max_augmented_output_readout.py
pytest -q

GPU_ID=0 SEED=1234 RUN_NAME="v74-collision-c2-mean-visit_taskgrad_half_d-32-p-profile-s1234" TRAIN_STEPS_OVERRIDE=100 bash scripts/train_v74_collision_c2_mean_visit_taskgrad_half_d_32_p.sh

for lr_tag in lr06 lr08 lr10; do
  case "${lr_tag}" in
    lr06) lr="0.6" ;;
    lr08) lr="0.8" ;;
    lr10) lr="1.0" ;;
  esac
  run_train 0 c2 mean p 1234 "${lr}" "${lr_tag}" &
  pid0=$!
  run_train 1 c2 maxaug p 1234 "${lr}" "${lr_tag}" &
  pid1=$!
  wait "${pid0}" "${pid1}"
done

python scripts/run_v74_eval_sweep.py --regimes c2 --conditions mean maxaug --schedules p --batches 48
python scripts/build_v74_report.py

MEAN_LR="$(python - <<'PY'
import json, pathlib
obj=json.loads(pathlib.Path('reports/summary_metrics_v74.json').read_text())
print(obj['pilot_choices']['mean'].get('lr_multiplier', 0.6))
PY
)"
MAXAUG_LR="$(python - <<'PY'
import json, pathlib
obj=json.loads(pathlib.Path('reports/summary_metrics_v74.json').read_text())
print(obj['pilot_choices']['maxaug'].get('lr_multiplier', 0.6))
PY
)"

for regime in c1 c2; do
  for seed in 2234 3234; do
    run_train 0 "${regime}" mean m "${seed}" "${MEAN_LR}" "" &
    pid0=$!
    run_train 1 "${regime}" maxaug m "${seed}" "${MAXAUG_LR}" "" &
    pid1=$!
    wait "${pid0}" "${pid1}"
  done
done

python scripts/run_v74_eval_sweep.py --regimes c1 c2 --conditions mean maxaug --schedules m --batches 96
python scripts/run_v74_eval_sweep.py --regimes c2 --conditions mean maxaug --schedules m --batches 96 --collect-probes
python scripts/build_v74_report.py

STRONGEST_REGIME="$(python - <<'PY'
import json, pathlib
obj=json.loads(pathlib.Path('reports/summary_metrics_v74.json').read_text())
print(obj.get('strongest_regime', 'c2'))
PY
)"

run_train 0 "${STRONGEST_REGIME}" mean l 4234 "${MEAN_LR}" "" &
pid0=$!
run_train 1 "${STRONGEST_REGIME}" maxaug l 4234 "${MAXAUG_LR}" "" &
pid1=$!
wait "${pid0}" "${pid1}"

python scripts/run_v74_eval_sweep.py --regimes "${STRONGEST_REGIME}" --conditions mean maxaug --schedules l --batches 128
python scripts/build_v74_report.py
pytest -q tests/test_v74_max_augmented_output_readout.py

bd update "${TASK_ID}" --append-notes "Completed v74 max-augmented output readout: kept the v68 mean-summary output readout and added a zero-init max-summary correction from the home cache, ran fair-shot pilots plus matched C1/C2 validation and a fresh rerun, rebuilt v74 summary/report, and reran focused verification."
bd close "${TASK_ID}"
