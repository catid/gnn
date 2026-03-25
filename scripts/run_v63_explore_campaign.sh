#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

ISSUE_ID="${1:-gnn-fuk}"
SUMMARY="reports/summary_metrics_v63_explore_exploit.json"

extract_json() {
  local code="$1"
  python3 - <<PY
import json
from pathlib import Path
obj = json.loads(Path("${SUMMARY}").read_text())
${code}
PY
}

arm_setting() {
  local arm="$1"
  local field="$2"
  python3 - "$arm" "$field" <<'PY'
import json, sys
from pathlib import Path
arm, field = sys.argv[1:3]
obj = json.loads(Path("reports/summary_metrics_v63_explore_exploit.json").read_text())
value = obj["selected_settings"][arm][field]
print(value)
PY
}

append_extra_arg() {
  local -n _arr="$1"
  local flag="$2"
  local value="$3"
  _arr+=("${flag}" "${value}")
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

build_runtime_args() {
  local arm="$1"
  local -a args=()
  local lr
  lr="$(arm_setting "${arm}" lr_multiplier)"
  append_extra_arg args --lr-scale "${lr}"

  local p_keep
  p_keep="$(arm_setting "${arm}" p_keep_prev)"
  if python3 - <<PY >/dev/null
value=float("${p_keep}")
raise SystemExit(0 if value > 0 else 1)
PY
  then
    append_extra_arg args --contract-penultimate-keep-prob "${p_keep}"
  fi

  local stability
  stability="$(arm_setting "${arm}" stability_weight)"
  if python3 - <<PY >/dev/null
value=float("${stability}")
raise SystemExit(0 if value > 0 else 1)
PY
  then
    append_extra_arg args --late-stage-stability-weight "${stability}"
  fi

  local slow_commit
  slow_commit="$(arm_setting "${arm}" slow_commit_interval)"
  if [[ "${slow_commit}" != "0" && "${slow_commit}" != "1" ]]; then
    append_extra_arg args --slow-commit-interval "${slow_commit}"
  fi

  local online_stage
  online_stage="$(arm_setting "${arm}" gate_online_stage_index_min)"
  if [[ "${online_stage}" != "-1" ]]; then
    append_extra_arg args --selector-gate-online-stage-index-min "${online_stage}"
    append_extra_arg args --selector-gate-online-entropy-high-threshold "$(arm_setting "${arm}" gate_online_entropy_high_threshold)"
    append_extra_arg args --selector-gate-online-gini-high-threshold "$(arm_setting "${arm}" gate_online_gini_high_threshold)"
  fi

  printf '%s\n' "${args[*]}"
}

run_train() {
  local gpu_id="$1"
  local regime="$2"
  local arm="$3"
  local schedule="$4"
  local seed="$5"
  local extra_args="$6"
  local tag="${7:-}"

  local run_name="v63ee-${regime}-${arm}-32-${schedule}"
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
  bash "scripts/train_v63ee_${regime}_${arm}_32_${schedule}.sh"
}

pilot_exploit() {
  local gpu_id="$1"
  shift
  local arms=("$@")
  for regime in core t1; do
    for arm in "${arms[@]}"; do
      for lr in 0.6 0.8 1.0; do
        if [[ "${arm}" == *_ds || "${arm}" == *_dsg ]]; then
          for keep_prob in 0.10 0.25; do
            local tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr*10)))
PY
)")p$(printf '%03d' "$(python3 - <<PY
p=${keep_prob}
print(int(round(p*100)))
PY
)")"
            run_train "${gpu_id}" "${regime}" "${arm}" p 1234 "--lr-scale ${lr} --contract-penultimate-keep-prob ${keep_prob}" "${tag}"
          done
        else
          local tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr*10)))
PY
)")"
          run_train "${gpu_id}" "${regime}" "${arm}" p 1234 "--lr-scale ${lr}" "${tag}"
        fi
      done
    done
  done
}

pilot_explore() {
  local gpu_id="$1"
  shift
  local arms=("$@")
  for regime in core t1; do
    for arm in "${arms[@]}"; do
      for lr in 0.6 0.8 1.0; do
        local keep_values=(0.10 0.25)
        local stability_values=(0.0)
        local online_templates=("5 0.80 0.55" "5 0.70 0.50")
        local commit_values=(0)
        if [[ "${arm}" == stable_* ]]; then
          stability_values=(0.005 0.010)
        fi
        if [[ "${arm}" == "gonline" ]]; then
          :
        else
          online_templates=("-1 0 0")
        fi
        if [[ "${arm}" == "slowcommit" ]]; then
          commit_values=(2 3)
        fi
        for keep_prob in "${keep_values[@]}"; do
          for stability in "${stability_values[@]}"; do
            for online_tpl in "${online_templates[@]}"; do
              set -- ${online_tpl}
              local online_stage="$1"
              local online_entropy="$2"
              local online_gini="$3"
              for commit_interval in "${commit_values[@]}"; do
                local tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr*10)))
PY
)")p$(printf '%03d' "$(python3 - <<PY
p=${keep_prob}
print(int(round(p*100)))
PY
)")"
                local extra="--lr-scale ${lr} --contract-penultimate-keep-prob ${keep_prob}"
                if python3 - <<PY >/dev/null
value=float("${stability}")
raise SystemExit(0 if value > 0 else 1)
PY
                then
                  tag="${tag}lam$(printf '%03d' "$(python3 - <<PY
lam=${stability}
print(int(round(lam*1000)))
PY
)")"
                  extra="${extra} --late-stage-stability-weight ${stability}"
                fi
                if [[ "${arm}" == "gonline" ]]; then
                  local entropy_tag gini_tag
                  entropy_tag="$(python3 - <<PY
value=float("${online_entropy}")
print(int(round(value * 100)))
PY
)"
                  gini_tag="$(python3 - <<PY
value=float("${online_gini}")
print(int(round(value * 100)))
PY
)"
                  tag="${tag}g$(printf '%02d' "${online_stage}")e$(printf '%02d' "${entropy_tag}")n$(printf '%02d' "${gini_tag}")"
                  extra="${extra} --selector-gate-online-stage-index-min ${online_stage} --selector-gate-online-entropy-high-threshold ${online_entropy} --selector-gate-online-gini-high-threshold ${online_gini}"
                fi
                if [[ "${arm}" == "slowcommit" ]]; then
                  tag="${tag}c${commit_interval}"
                  extra="${extra} --slow-commit-interval ${commit_interval}"
                fi
                run_train "${gpu_id}" "${regime}" "${arm}" p 1234 "${extra}" "${tag}"
              done
            done
          done
        done
      done
    done
  done
}

run_arm_schedule() {
  local gpu_id="$1"
  local schedule="$2"
  local seed_list="$3"
  local regime_list="$4"
  shift 4
  local arms=("$@")
  # shellcheck disable=SC2206
  local seeds=(${seed_list})
  # shellcheck disable=SC2206
  local regimes=(${regime_list})

  for arm in "${arms[@]}"; do
    local extra_args
    extra_args="$(build_runtime_args "${arm}")"
    for regime in "${regimes[@]}"; do
      for seed in "${seeds[@]}"; do
        run_train "${gpu_id}" "${regime}" "${arm}" "${schedule}" "${seed}" "${extra_args}"
      done
    done
  done
}

run_arm_schedule_tagged() {
  local gpu_id="$1"
  local schedule="$2"
  local seed="$3"
  local tag="$4"
  local regime_list="$5"
  shift 5
  local arms=("$@")
  # shellcheck disable=SC2206
  local regimes=(${regime_list})
  for arm in "${arms[@]}"; do
    local extra_args
    extra_args="$(build_runtime_args "${arm}")"
    for regime in "${regimes[@]}"; do
      run_train "${gpu_id}" "${regime}" "${arm}" "${schedule}" "${seed}" "${extra_args}" "${tag}"
    done
  done
}

python scripts/gen_v63_explore_configs.py
python -m compileall apsgnn scripts/gen_v63_explore_configs.py scripts/run_v63_explore_eval_sweep.py scripts/build_v63_explore_report.py tests/test_v63_explore_exploit_balance.py
pytest -q tests/test_v63_explore_exploit_balance.py
pytest -q

GPU_ID=0 RUN_NAME=v63ee-calibration-core-visitonly_b-32-p-100-s1234 TRAIN_STEPS_OVERRIDE=100 TRAIN_LOG_TO_FILE=1 TRAIN_LOG_PATH=runs/v63ee-calibration-core-visitonly_b-32-p-100-s1234.console.log bash scripts/train_v63ee_core_visitonly_b_32_p.sh

(
  pilot_exploit 0 visitonly_b visitonly_d visitonly_ds visitonly_dsg
  pilot_explore 0 stage_late_vt stable_v gonline
) &
PID_PILOT_0=$!
(
  pilot_exploit 1 visit_taskgrad_half_b visit_taskgrad_half_d visit_taskgrad_half_ds visit_taskgrad_half_dsg
  pilot_explore 1 stage_early_vt stable_vt slowcommit
) &
PID_PILOT_1=$!
wait "${PID_PILOT_0}" "${PID_PILOT_1}"

python scripts/run_v63_explore_eval_sweep.py --regimes core t1 --schedules p --arms \
  visitonly_b visit_taskgrad_half_b visitonly_d visit_taskgrad_half_d visitonly_ds visit_taskgrad_half_ds visitonly_dsg visit_taskgrad_half_dsg \
  stage_late_vt stage_early_vt stable_v stable_vt gonline slowcommit \
  --batches 64 --best-only
python scripts/build_v63_explore_report.py

(
  run_arm_schedule 0 s "1234" "core t1 t2a" visitonly_b visitonly_d visitonly_ds visitonly_dsg
) &
PID_EXPLOIT_0=$!
(
  run_arm_schedule 1 s "1234" "core t1 t2a" visit_taskgrad_half_b visit_taskgrad_half_d visit_taskgrad_half_ds visit_taskgrad_half_dsg
) &
PID_EXPLOIT_1=$!
wait "${PID_EXPLOIT_0}" "${PID_EXPLOIT_1}"

(
  run_arm_schedule 0 s "1234" "core t2a" stage_late_vt stage_early_vt stable_v stable_vt gonline
) &
PID_EXPLORE_E1_0=$!
(
  run_arm_schedule 1 s "1234" "core hmix" slowcommit
) &
PID_EXPLORE_E1_1=$!
wait "${PID_EXPLORE_E1_0}" "${PID_EXPLORE_E1_1}"

python scripts/run_v63_explore_eval_sweep.py --regimes core t1 t2a --schedules s --arms \
  visitonly_b visit_taskgrad_half_b visitonly_d visit_taskgrad_half_d visitonly_ds visit_taskgrad_half_ds visitonly_dsg visit_taskgrad_half_dsg \
  --batches 96
python scripts/run_v63_explore_eval_sweep.py --regimes core t2a hmix --schedules s --arms \
  stage_late_vt stage_early_vt stable_v stable_vt gonline slowcommit \
  --batches 96
python scripts/build_v63_explore_report.py

mapfile -t promoted_exploration < <(extract_json 'print("\n".join(obj["promoted_exploration"]))')
MID=$(( (${#promoted_exploration[@]} + 1) / 2 ))
ARMS_0=("${promoted_exploration[@]:0:${MID}}")
ARMS_1=("${promoted_exploration[@]:${MID}}")

(
  run_arm_schedule 0 m "2234 3234" "t1 hmix" "${ARMS_0[@]}"
) &
PID_E2_0=$!
(
  run_arm_schedule 1 m "2234 3234" "t1 hmix" "${ARMS_1[@]}"
) &
PID_E2_1=$!
wait "${PID_E2_0}" "${PID_E2_1}"

python scripts/run_v63_explore_eval_sweep.py --regimes t1 hmix --schedules m --arms "${promoted_exploration[@]}" --batches 96
python scripts/build_v63_explore_report.py

mapfile -t finalists < <(extract_json 'print("\n".join(obj["finalists_shared_verification"]))')
MID=$(( (${#finalists[@]} + 1) / 2 ))
FINAL_0=("${finalists[@]:0:${MID}}")
FINAL_1=("${finalists[@]:${MID}}")

(
  run_arm_schedule 0 m "4234" "t1r t2b t2c hmid" "${FINAL_0[@]}"
) &
PID_SHARED_0=$!
(
  run_arm_schedule 1 m "4234" "t1r t2b t2c hmid" "${FINAL_1[@]}"
) &
PID_SHARED_1=$!
wait "${PID_SHARED_0}" "${PID_SHARED_1}"

python scripts/run_v63_explore_eval_sweep.py --regimes t1r t2b t2c hmid --schedules m --arms "${finalists[@]}" --batches 128
python scripts/build_v63_explore_report.py

mapfile -t final_candidates < <(extract_json 'print("\n".join(obj["final_candidates"]))')
mapfile -t rerun_regimes < <(extract_json 'print("\n".join(obj["final_rerun_regimes"]))')

(
  run_arm_schedule_tagged 0 l "5234" "rerun" "${rerun_regimes[0]} ${rerun_regimes[1]}" "${final_candidates[0]}"
) &
PID_RERUN_0=$!
(
  run_arm_schedule_tagged 1 l "5234" "rerun" "${rerun_regimes[0]} ${rerun_regimes[1]}" "${final_candidates[1]}"
) &
PID_RERUN_1=$!
wait "${PID_RERUN_0}" "${PID_RERUN_1}"

python scripts/run_v63_explore_eval_sweep.py --regimes "${rerun_regimes[@]}" --schedules l --arms "${final_candidates[@]}" --tags rerun --batches 160
python scripts/run_v63_explore_eval_sweep.py --regimes core t1 t2a hmix --schedules s m l --arms "${final_candidates[@]}" --batches 128 --extra-depth --settle-cap 30 --surface
python scripts/build_v63_explore_report.py

OUTCOME="$(extract_json 'print(obj["outcome"]["outcome"])')"
if [[ "${OUTCOME}" == "unresolved" ]]; then
  AMBIG_REGIME="${rerun_regimes[0]}"
  run_arm_schedule_tagged 0 l "6234" "amb" "${AMBIG_REGIME}" "${final_candidates[0]}"
  run_arm_schedule_tagged 1 l "6234" "amb" "${AMBIG_REGIME}" "${final_candidates[1]}"
  python scripts/run_v63_explore_eval_sweep.py --regimes "${AMBIG_REGIME}" --schedules l --arms "${final_candidates[@]}" --tags amb --batches 160
  python scripts/build_v63_explore_report.py
fi

pytest -q
bd update "${ISSUE_ID}" --append-notes "Completed the broadened v63 explore/exploit contracts campaign with pilots, exploit and exploration matrices, shared verification, reruns, extra-depth settling evals, surface maps, and ambiguity-breaker if needed."
bd close "${ISSUE_ID}"
