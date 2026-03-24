#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

ISSUE_ID="${1:-gnn-7w7}"

extract_json() {
  local code="$1"
  python3 - <<PY
import json
from pathlib import Path
obj = json.loads(Path("reports/summary_metrics_v63.json").read_text())
${code}
PY
}

pair_setting() {
  local pair="$1"
  local field="$2"
  python3 - "$pair" "$field" <<'PY'
import json, sys
from pathlib import Path
pair, field = sys.argv[1:3]
obj = json.loads(Path("reports/summary_metrics_v63.json").read_text())
value = obj["selected_settings"][pair][field]
print(value)
PY
}

run_train() {
  local gpu_id="$1"
  local regime="$2"
  local pair="$3"
  local schedule="$4"
  local seed="$5"
  local lr_scale="$6"
  local keep_prob="$7"
  local tag="${8:-}"

  local run_name="v63-${regime}-${pair}-32-${schedule}"
  if [[ -n "${tag}" ]]; then
    run_name="${run_name}-${tag}"
  fi
  run_name="${run_name}-s${seed}"

  export GPU_ID="${gpu_id}"
  export SEED="${seed}"
  export RUN_NAME="${run_name}"
  export TRAIN_LOG_TO_FILE=1
  export TRAIN_LOG_PATH="runs/${run_name}.console.log"
  if [[ "${keep_prob}" == "na" ]]; then
    export EXTRA_ARGS="--lr-scale ${lr_scale}"
  else
    export EXTRA_ARGS="--lr-scale ${lr_scale} --contract-penultimate-keep-prob ${keep_prob}"
  fi
  bash "scripts/train_v63_${regime}_${pair}_32_${schedule}.sh"
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
    local keep_prob
    keep_prob="$(pair_setting "${pair}" p_keep_prev)"
    if [[ "${pair}" != *_ds && "${pair}" != *_dsg ]]; then
      keep_prob="na"
    fi
    for regime in "${regimes[@]}"; do
      for seed in "${seeds[@]}"; do
        run_train "${gpu_id}" "${regime}" "${pair}" "${schedule}" "${seed}" "${lr_scale}" "${keep_prob}"
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
    local keep_prob
    keep_prob="$(pair_setting "${pair}" p_keep_prev)"
    if [[ "${pair}" != *_ds && "${pair}" != *_dsg ]]; then
      keep_prob="na"
    fi
    for regime in "${regimes[@]}"; do
      run_train "${gpu_id}" "${regime}" "${pair}" "${schedule}" "${seed}" "${lr_scale}" "${keep_prob}" "${tag}"
    done
  done
}

python scripts/gen_v63_configs.py
python -m compileall apsgnn scripts/gen_v63_configs.py scripts/run_v63_eval_sweep.py scripts/build_v63_report.py tests/test_v63_rsm_lite_contracts.py
pytest -q tests/test_v63_rsm_lite_contracts.py
pytest -q

GPU_ID=0 RUN_NAME=v63-calibration-core-visitonly_b-32-p-100-s1234 TRAIN_STEPS_OVERRIDE=100 TRAIN_LOG_TO_FILE=1 TRAIN_LOG_PATH=runs/v63-calibration-core-visitonly_b-32-p-100-s1234.console.log bash scripts/train_v63_core_visitonly_b_32_p.sh

(
  for regime in core t1; do
    for pair in visitonly_b visitonly_d visitonly_ds visitonly_dsg; do
      for lr in 0.6 0.8 1.0; do
        if [[ "${pair}" == *_ds || "${pair}" == *_dsg ]]; then
          for keep_prob in 0.10 0.25; do
            tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr*10)))
PY
)")p$(printf '%03d' "$(python3 - <<PY
p=${keep_prob}
print(int(round(p*100)))
PY
)")"
            run_train 0 "${regime}" "${pair}" p 1234 "${lr}" "${keep_prob}" "${tag}"
          done
        else
          tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr*10)))
PY
)")"
          run_train 0 "${regime}" "${pair}" p 1234 "${lr}" "na" "${tag}"
        fi
      done
    done
  done
) &
PID_PILOT_0=$!
(
  for regime in core t1; do
    for pair in visit_taskgrad_half_b visit_taskgrad_half_d visit_taskgrad_half_ds visit_taskgrad_half_dsg; do
      for lr in 0.6 0.8 1.0; do
        if [[ "${pair}" == *_ds || "${pair}" == *_dsg ]]; then
          for keep_prob in 0.10 0.25; do
            tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr*10)))
PY
)")p$(printf '%03d' "$(python3 - <<PY
p=${keep_prob}
print(int(round(p*100)))
PY
)")"
            run_train 1 "${regime}" "${pair}" p 1234 "${lr}" "${keep_prob}" "${tag}"
          done
        else
          tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr*10)))
PY
)")"
          run_train 1 "${regime}" "${pair}" p 1234 "${lr}" "na" "${tag}"
        fi
      done
    done
  done
) &
PID_PILOT_1=$!
wait "${PID_PILOT_0}" "${PID_PILOT_1}"

python scripts/run_v63_eval_sweep.py --regimes core t1 --schedules p --pairs visitonly_b visitonly_d visitonly_ds visitonly_dsg visit_taskgrad_half_b visit_taskgrad_half_d visit_taskgrad_half_ds visit_taskgrad_half_dsg --batches 64 --best-only
python scripts/build_v63_report.py

(
  run_pair_schedule 0 s "1234" "core t1 t2a" visitonly_b visitonly_d visitonly_ds visitonly_dsg
) &
PID_S_0=$!
(
  run_pair_schedule 1 s "1234" "core t1 t2a" visit_taskgrad_half_b visit_taskgrad_half_d visit_taskgrad_half_ds visit_taskgrad_half_dsg
) &
PID_S_1=$!
wait "${PID_S_0}" "${PID_S_1}"

python scripts/run_v63_eval_sweep.py --regimes core t1 t2a --schedules s --pairs visitonly_b visitonly_d visitonly_ds visitonly_dsg visit_taskgrad_half_b visit_taskgrad_half_d visit_taskgrad_half_ds visit_taskgrad_half_dsg --batches 96
python scripts/build_v63_report.py

mapfile -t promoted_pairs < <(extract_json 'print("\n".join(obj["promoted_pairs"]))')
MID=$(( (${#promoted_pairs[@]} + 1) / 2 ))
PAIRS_0=("${promoted_pairs[@]:0:${MID}}")
PAIRS_1=("${promoted_pairs[@]:${MID}}")

(
  run_pair_schedule 0 m "2234 3234" "core t1 t2a" "${PAIRS_0[@]}"
) &
PID_M_0=$!
(
  run_pair_schedule 1 m "2234 3234" "core t1 t2a" "${PAIRS_1[@]}"
) &
PID_M_1=$!
wait "${PID_M_0}" "${PID_M_1}"

python scripts/run_v63_eval_sweep.py --regimes core t1 t2a --schedules m --pairs "${promoted_pairs[@]}" --batches 96
python scripts/build_v63_report.py

mapfile -t top2_pairs < <(extract_json 'print("\n".join(obj["top2_pairs_after_confirmation"]))')
(
  run_pair_schedule 0 l "4234" "t1r t2b t2c hmid hmix" "${top2_pairs[0]}"
) &
PID_L_0=$!
(
  run_pair_schedule 1 l "4234" "t1r t2b t2c hmid hmix" "${top2_pairs[1]}"
) &
PID_L_1=$!
wait "${PID_L_0}" "${PID_L_1}"

python scripts/run_v63_eval_sweep.py --regimes t1r t2b t2c hmid hmix --schedules l --pairs "${top2_pairs[@]}" --batches 128
python scripts/build_v63_report.py

mapfile -t final_pairs < <(extract_json 'print("\n".join(obj["top2_pairs_after_holdout"]))')
python scripts/run_v63_eval_sweep.py --regimes core t1 t2a --schedules m --pairs "${final_pairs[@]}" --batches 128 --extra-depth --settle-cap 30
python scripts/run_v63_eval_sweep.py --regimes hmix --schedules l --pairs "${final_pairs[@]}" --batches 128 --extra-depth --settle-cap 30
python scripts/build_v63_report.py

mapfile -t rerun_regimes < <(extract_json 'print("\n".join(obj["final_rerun_regimes"]))')
(
  run_pair_schedule_tagged 0 l "5234" "rerun" "${rerun_regimes[0]} ${rerun_regimes[1]}" "${final_pairs[0]}"
) &
PID_R_0=$!
(
  run_pair_schedule_tagged 1 l "5234" "rerun" "${rerun_regimes[0]} ${rerun_regimes[1]}" "${final_pairs[1]}"
) &
PID_R_1=$!
wait "${PID_R_0}" "${PID_R_1}"

python scripts/run_v63_eval_sweep.py --regimes "${rerun_regimes[@]}" --schedules l --pairs "${final_pairs[@]}" --tags rerun --batches 160
python scripts/build_v63_report.py

OUTCOME="$(extract_json 'print(obj["outcome"]["outcome"])')"
if [[ "${OUTCOME}" == "unresolved" ]]; then
  AMBIG_REGIME="${rerun_regimes[0]}"
  run_pair_schedule_tagged 0 l "6234" "amb" "${AMBIG_REGIME}" "${final_pairs[0]}"
  run_pair_schedule_tagged 1 l "6234" "amb" "${AMBIG_REGIME}" "${final_pairs[1]}"
  python scripts/run_v63_eval_sweep.py --regimes "${AMBIG_REGIME}" --schedules l --pairs "${final_pairs[@]}" --tags amb --batches 160
  python scripts/build_v63_report.py
fi

pytest -q
bd update "${ISSUE_ID}" --append-notes "Completed v63 RSM-lite contract campaign with pilots, screening, confirmation, holdouts, extra-depth settling evals, reruns, and ambiguity-breaker if needed."
bd close "${ISSUE_ID}"
