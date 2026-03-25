#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

ISSUE_ID="${1:-gnn-3wp}"
SUMMARY="reports/summary_metrics_v64.json"

extract_json() {
  local code="$1"
  python3 - <<PY
import json
from pathlib import Path
obj = json.loads(Path("${SUMMARY}").read_text())
${code}
PY
}

pair_setting() {
  local pair="$1"
  local field="$2"
  python3 - "$pair" "$field" <<'PY'
import json
import sys
from pathlib import Path

pair, field = sys.argv[1:3]
obj = json.loads(Path("reports/summary_metrics_v64.json").read_text())
value = obj["selected_settings"][pair][field]
print(value)
PY
}

have_completed_run() {
  local run_name="$1"
  python3 - "$run_name" <<'PY'
import json
import sys
from pathlib import Path

import yaml

run_name = sys.argv[1]
for candidate in sorted(Path("runs").glob(f"*-{run_name}")):
    if not candidate.is_dir():
        continue
    metrics_path = candidate / "metrics.jsonl"
    config_path = candidate / "config.yaml"
    last_path = candidate / "last.pt"
    if not metrics_path.exists() or not config_path.exists() or not last_path.exists():
        continue
    try:
        config = yaml.safe_load(config_path.read_text())
        expected = int(config.get("train", {}).get("train_steps", 0))
    except Exception:
        continue
    if expected <= 0:
        continue
    max_step = 0
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            max_step = max(max_step, int(json.loads(line).get("step", 0)))
        except Exception:
            continue
    if max_step >= expected:
        print(candidate)
        raise SystemExit(0)
raise SystemExit(1)
PY
}

run_train() {
  local gpu_id="$1"
  local regime="$2"
  local pair="$3"
  local schedule="$4"
  local seed="$5"
  local extra_args="$6"
  local tag="${7:-}"

  local run_name="v64-${regime}-${pair}-32-${schedule}"
  if [[ -n "${tag}" ]]; then
    run_name="${run_name}-${tag}"
  fi
  run_name="${run_name}-s${seed}"

  local completed_dir
  if completed_dir="$(have_completed_run "${run_name}")"; then
    echo "skip ${run_name} using ${completed_dir}"
    return 0
  fi

  export GPU_ID="${gpu_id}"
  export SEED="${seed}"
  export RUN_NAME="${run_name}"
  export TRAIN_LOG_TO_FILE=1
  export TRAIN_LOG_PATH="runs/${run_name}.console.log"
  export EXTRA_ARGS="${extra_args}"
  bash "scripts/train_v64_${regime}_${pair}_32_${schedule}.sh"
}

run_pair_schedule() {
  local gpu_id="$1"
  local schedule="$2"
  local seed_list="$3"
  local regime_list="$4"
  shift 4
  local pairs=("$@")
  # shellcheck disable=SC2206
  local seeds=(${seed_list})
  # shellcheck disable=SC2206
  local regimes=(${regime_list})
  for pair in "${pairs[@]}"; do
    local lr_scale
    lr_scale="$(pair_setting "${pair}" lr_multiplier)"
    for regime in "${regimes[@]}"; do
      for seed in "${seeds[@]}"; do
        run_train "${gpu_id}" "${regime}" "${pair}" "${schedule}" "${seed}" "--lr-scale ${lr_scale}"
      done
    done
  done
}

run_pair_schedule_tagged() {
  local gpu_id="$1"
  local schedule="$2"
  local seed="$3"
  local tag="$4"
  local regime_list="$5"
  shift 5
  local pairs=("$@")
  # shellcheck disable=SC2206
  local regimes=(${regime_list})
  for pair in "${pairs[@]}"; do
    local lr_scale
    lr_scale="$(pair_setting "${pair}" lr_multiplier)"
    for regime in "${regimes[@]}"; do
      run_train "${gpu_id}" "${regime}" "${pair}" "${schedule}" "${seed}" "--lr-scale ${lr_scale}" "${tag}"
    done
  done
}

python scripts/gen_v64_configs.py
python -m compileall apsgnn scripts/gen_v64_configs.py scripts/run_v64_eval_sweep.py scripts/build_v64_report.py tests/test_v64_ds_contract_factorization.py
pytest -q tests/test_v64_ds_contract_factorization.py
pytest -q

GPU_ID=0 RUN_NAME=v64-calibration-core-visitonly-b-32-p-100-s1234 TRAIN_STEPS_OVERRIDE=100 TRAIN_LOG_TO_FILE=1 TRAIN_LOG_PATH=runs/v64-calibration-core-visitonly-b-32-p-100-s1234.console.log bash scripts/train_v63_core_visitonly_b_32_p.sh

(
  for regime in core t1; do
    for pair in visitonly_d visitonly_ds_p005 visitonly_ds_p010 visitonly_ds_p025; do
      for lr in 0.6 0.8 1.0; do
        tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
        run_train 0 "${regime}" "${pair}" p 1234 "--lr-scale ${lr}" "${tag}"
      done
    done
  done
) &
PID_CORE_0=$!
(
  for regime in core t1; do
    for pair in visitonly_ds_p040 visitonly_ds_fixed1step visitonly_ds_fixed2step; do
      for lr in 0.6 0.8 1.0; do
        tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
        run_train 1 "${regime}" "${pair}" p 1234 "--lr-scale ${lr}" "${tag}"
      done
    done
  done
) &
PID_CORE_1=$!
wait "${PID_CORE_0}" "${PID_CORE_1}"

python scripts/run_v64_eval_sweep.py --regimes core t1 --schedules p --pairs \
  visitonly_d visitonly_ds_p005 visitonly_ds_p010 visitonly_ds_p025 visitonly_ds_p040 visitonly_ds_fixed1step visitonly_ds_fixed2step \
  --batches 64 --best-only
python scripts/build_v64_report.py

CORE_BEST="$(extract_json 'print(obj["selected_core_contracts"]["best"] or "")')"
CORE_RUNNER="$(extract_json 'print(obj["selected_core_contracts"]["runner_up"] or "")')"
python scripts/gen_v64_configs.py --core-best "${CORE_BEST}" --core-runner-up "${CORE_RUNNER}" --aux-final-multiplier 0.50

(
  for regime in core t1; do
    for pair in visitonly_d visitonly_ds_core_best visitonly_ds_core_runner_up visitonly_ds_auxanneal_050 visitonly_ds_auxanneal_025 visitonly_ds_randdepth; do
      for lr in 0.6 0.8 1.0; do
        tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
        run_train 0 "${regime}" "${pair}" p 1234 "--lr-scale ${lr}" "${tag}"
      done
    done
  done
) &
PID_PILOT_0=$!
(
  for regime in core t1; do
    for pair in visit_taskgrad_half_d visit_taskgrad_half_ds_core_best visit_taskgrad_half_ds_core_runner_up visit_taskgrad_half_ds_auxanneal_050 visit_taskgrad_half_ds_auxanneal_025 visit_taskgrad_half_ds_randdepth; do
      for lr in 0.6 0.8 1.0; do
        tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
        run_train 1 "${regime}" "${pair}" p 1234 "--lr-scale ${lr}" "${tag}"
      done
    done
  done
) &
PID_PILOT_1=$!
wait "${PID_PILOT_0}" "${PID_PILOT_1}"

python scripts/run_v64_eval_sweep.py --regimes core t1 --schedules p --pairs \
  visitonly_d visit_taskgrad_half_d visitonly_ds_core_best visit_taskgrad_half_ds_core_best visitonly_ds_core_runner_up visit_taskgrad_half_ds_core_runner_up \
  visitonly_ds_auxanneal_050 visit_taskgrad_half_ds_auxanneal_050 visitonly_ds_auxanneal_025 visit_taskgrad_half_ds_auxanneal_025 \
  visitonly_ds_randdepth visit_taskgrad_half_ds_randdepth \
  --batches 64 --best-only
python scripts/build_v64_report.py

AUX_FINAL="$(extract_json 'print(format(obj.get("chosen_aux_anneal_final_multiplier", 0.5), ".2f"))')"
python scripts/gen_v64_configs.py --core-best "${CORE_BEST}" --core-runner-up "${CORE_RUNNER}" --aux-final-multiplier "${AUX_FINAL}"

(
  run_pair_schedule 0 s "1234" "core t1 t2a" \
    visitonly_d visitonly_ds_core_best visitonly_ds_core_runner_up visitonly_ds_auxanneal visitonly_ds_randdepth
) &
PID_SCREEN_0=$!
(
  run_pair_schedule 1 s "1234" "core t1 t2a" \
    visit_taskgrad_half_d visit_taskgrad_half_ds_core_best visit_taskgrad_half_ds_core_runner_up visit_taskgrad_half_ds_auxanneal visit_taskgrad_half_ds_randdepth
) &
PID_SCREEN_1=$!
wait "${PID_SCREEN_0}" "${PID_SCREEN_1}"

python scripts/run_v64_eval_sweep.py --regimes core t1 t2a --schedules s --pairs \
  visitonly_d visit_taskgrad_half_d visitonly_ds_core_best visit_taskgrad_half_ds_core_best visitonly_ds_core_runner_up visit_taskgrad_half_ds_core_runner_up \
  visitonly_ds_auxanneal visit_taskgrad_half_ds_auxanneal visitonly_ds_randdepth visit_taskgrad_half_ds_randdepth \
  --batches 96
python scripts/build_v64_report.py

mapfile -t HMIX_CONTRACTS < <(extract_json 'print("\n".join(obj["promoted_contracts_screening"]))')
(
  for contract in "${HMIX_CONTRACTS[@]}"; do
    run_pair_schedule 0 s "1234" "hmix" "visitonly_${contract}"
  done
) &
PID_HMIX_0=$!
(
  for contract in "${HMIX_CONTRACTS[@]}"; do
    run_pair_schedule 1 s "1234" "hmix" "visit_taskgrad_half_${contract}"
  done
) &
PID_HMIX_1=$!
wait "${PID_HMIX_0}" "${PID_HMIX_1}"

HMIX_PAIR_ARGS=()
for contract in "${HMIX_CONTRACTS[@]}"; do
  HMIX_PAIR_ARGS+=("visitonly_${contract}" "visit_taskgrad_half_${contract}")
done
python scripts/run_v64_eval_sweep.py --regimes hmix --schedules s --pairs "${HMIX_PAIR_ARGS[@]}" --batches 96
python scripts/build_v64_report.py

mapfile -t TOP2_CONTRACTS < <(extract_json 'print("\n".join(obj["top2_contracts_after_hmix"]))')
(
  run_pair_schedule 0 m "2234 3234" "core t1 t2a hmix" \
    "visitonly_${TOP2_CONTRACTS[0]}" "visitonly_${TOP2_CONTRACTS[1]}"
) &
PID_CONF_0=$!
(
  run_pair_schedule 1 m "2234 3234" "core t1 t2a hmix" \
    "visit_taskgrad_half_${TOP2_CONTRACTS[0]}" "visit_taskgrad_half_${TOP2_CONTRACTS[1]}"
) &
PID_CONF_1=$!
wait "${PID_CONF_0}" "${PID_CONF_1}"

CONFIRM_PAIR_ARGS=(
  "visitonly_${TOP2_CONTRACTS[0]}" "visit_taskgrad_half_${TOP2_CONTRACTS[0]}"
  "visitonly_${TOP2_CONTRACTS[1]}" "visit_taskgrad_half_${TOP2_CONTRACTS[1]}"
)
python scripts/run_v64_eval_sweep.py --regimes core t1 t2a hmix --schedules m --pairs "${CONFIRM_PAIR_ARGS[@]}" --batches 96
python scripts/build_v64_report.py

mapfile -t TOP2_PAIRS < <(extract_json 'print("\n".join(obj["top2_pairs_after_confirmation"]))')
(
  run_pair_schedule 0 l "4234" "t1r t2b t2c hmid" "${TOP2_PAIRS[0]}"
) &
PID_HOLD_0=$!
(
  run_pair_schedule 1 l "4234" "t1r t2b t2c hmid" "${TOP2_PAIRS[1]}"
) &
PID_HOLD_1=$!
wait "${PID_HOLD_0}" "${PID_HOLD_1}"

python scripts/run_v64_eval_sweep.py --regimes t1r t2b t2c hmid --schedules l --pairs "${TOP2_PAIRS[@]}" --batches 128
python scripts/build_v64_report.py

mapfile -t FINAL_PAIRS < <(extract_json 'print("\n".join(obj["top2_pairs_after_holdout"]))')
python scripts/run_v64_eval_sweep.py --regimes core t1 t2a hmix --schedules m --pairs "${FINAL_PAIRS[@]}" --batches 128 --extra-depth --settle-cap 24
python scripts/build_v64_report.py

mapfile -t RERUN_REGIMES < <(extract_json 'print("\n".join(obj["final_rerun_regimes"]))')
(
  run_pair_schedule_tagged 0 l "5234" "rerun" "${RERUN_REGIMES[0]} ${RERUN_REGIMES[1]}" "${FINAL_PAIRS[0]}"
) &
PID_RERUN_0=$!
(
  run_pair_schedule_tagged 1 l "5234" "rerun" "${RERUN_REGIMES[0]} ${RERUN_REGIMES[1]}" "${FINAL_PAIRS[1]}"
) &
PID_RERUN_1=$!
wait "${PID_RERUN_0}" "${PID_RERUN_1}"

python scripts/run_v64_eval_sweep.py --regimes "${RERUN_REGIMES[@]}" --schedules l --pairs "${FINAL_PAIRS[@]}" --tags rerun --batches 160
python scripts/build_v64_report.py

OUTCOME="$(extract_json 'print(obj["outcome"]["outcome"])')"
if [[ "${OUTCOME}" == "unresolved" ]]; then
  AMBIG_REGIME="${RERUN_REGIMES[0]}"
  run_pair_schedule_tagged 0 l "6234" "amb" "${AMBIG_REGIME}" "${FINAL_PAIRS[0]}"
  run_pair_schedule_tagged 1 l "6234" "amb" "${AMBIG_REGIME}" "${FINAL_PAIRS[1]}"
  python scripts/run_v64_eval_sweep.py --regimes "${AMBIG_REGIME}" --schedules l --pairs "${FINAL_PAIRS[@]}" --tags amb --batches 160
  python scripts/build_v64_report.py
fi

pytest -q
bd update "${ISSUE_ID}" --append-notes "Completed v64 DS factorization funnel with core pilots, screening, Hmix tiebreak, confirmation, holdouts, extra-depth evals, reruns, and ambiguity-breaker if needed."
bd close "${ISSUE_ID}"
