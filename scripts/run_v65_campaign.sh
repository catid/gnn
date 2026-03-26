#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

ISSUE_ID="${1:-gnn-8os}"
SUMMARY="reports/summary_metrics_v65.json"

summary_get() {
  local code="$1"
  python3 - <<PY
import json
from pathlib import Path
obj = json.loads(Path("${SUMMARY}").read_text())
${code}
PY
}

pilot_lr() {
  local family="$1"
  python3 - "$family" <<'PY'
import json
import sys
from pathlib import Path

family = sys.argv[1]
obj = json.loads(Path("reports/summary_metrics_v65.json").read_text())
choice = obj.get("pilot_choices", {}).get(family, {})
print(choice.get("lr_multiplier", 1.0))
PY
}

config_run_name() {
  local stem="$1"
  python3 - "$stem" <<'PY'
import sys
from pathlib import Path
import yaml

stem = sys.argv[1]
path = Path("configs") / f"{stem}.yaml"
cfg = yaml.safe_load(path.read_text())
print(cfg.get("runtime", {}).get("run_name", stem))
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
  local stem="$2"
  local seed="$3"
  local extra_args="$4"
  local tag="${5:-}"
  local base_run_name
  base_run_name="$(config_run_name "${stem}")"
  local run_name="${base_run_name}"
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
  bash "scripts/train_${stem}.sh"
}

python scripts/gen_v65_configs.py
python -m compileall apsgnn apsgnn/probes.py scripts/gen_v65_configs.py scripts/run_v65_eval_sweep.py scripts/build_v65_report.py tests/test_v65_architectural_headroom.py
pytest -q tests/test_v65_architectural_headroom.py
pytest -q

# 100-step calibration profile on the representative collision regime.
GPU_ID=0 RUN_NAME="v65-collision-c1-cacheon-classon-visit_taskgrad_half_d-32-p-profile-s1234" \
TRAIN_STEPS_OVERRIDE=100 TRAIN_LOG_TO_FILE=1 \
TRAIN_LOG_PATH="runs/v65-collision-c1-cacheon-classon-visit_taskgrad_half_d-32-p-profile-s1234.console.log" \
bash scripts/train_v65_collision_c1_cacheon_classon_visit_taskgrad_half_d_32_p.sh

# Sanity-gate LR pilots.
(
  for pair in visit_taskgrad_half_d visit_taskgrad_half_ds; do
    for lr in 0.6 0.8 1.0; do
      tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
      run_train 0 "v65_collision_c1_cacheon_classon_${pair}_32_p" 1234 "--lr-scale ${lr}" "${tag}"
    done
  done
) &
PID_SANITY_0=$!
(
  for pair in visitonly_d visitonly_ds; do
    for lr in 0.6 0.8 1.0; do
      tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
      run_train 1 "v65_collision_c1_cacheon_classon_${pair}_32_p" 1234 "--lr-scale ${lr}" "${tag}"
    done
  done
) &
PID_SANITY_1=$!
wait "${PID_SANITY_0}" "${PID_SANITY_1}"

python scripts/run_v65_eval_sweep.py --pack collision --regimes c1 --pairs visit_taskgrad_half_d visitonly_d visit_taskgrad_half_ds visitonly_ds --schedules p --conditions cacheon-classon --batches 64
python scripts/build_v65_report.py

for pair in visit_taskgrad_half_d visitonly_d visit_taskgrad_half_ds visitonly_ds; do
  lr="$(pilot_lr "collision|c1|cacheon-classon|${pair}")"
  run_train $([[ "${pair}" == visit_taskgrad_half* ]] && echo 0 || echo 1) "v65_collision_c1_cacheon_classon_${pair}_32_p" 1234 "--lr-scale ${lr}"
done

python scripts/run_v65_eval_sweep.py --pack collision --regimes c1 --pairs visit_taskgrad_half_d visitonly_d visit_taskgrad_half_ds visitonly_ds --schedules p --conditions cacheon-classon --batches 64
python scripts/build_v65_report.py

CHOSEN_CONTRACT="$(summary_get 'print(obj.get("chosen_contract_family", "d"))')"
INCUMBENT_PAIR="visit_taskgrad_half_${CHOSEN_CONTRACT}"
RUNNER_UP_PAIR="visitonly_${CHOSEN_CONTRACT}"
RUNNER_CACHE_LR="$(pilot_lr "collision|c1|cacheon-classon|${RUNNER_UP_PAIR}")"

# Targeted pilots for no-cache, class-slice-off, and delay controls.
(
  for lr in 0.6 0.8 1.0; do
    tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
    run_train 0 "v65_collision_c1_nocache_classon_${INCUMBENT_PAIR}_32_p" 1234 "--lr-scale ${lr}" "${tag}"
    run_train 0 "v65_collision_c1_cacheon_classoff_${INCUMBENT_PAIR}_32_p" 1234 "--lr-scale ${lr}" "${tag}"
  done
) &
PID_TARGET_0=$!
(
  for condition in learned zero random fixed; do
    for lr in 0.6 0.8 1.0; do
      tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
      run_train 1 "v65_delay_d1_${condition}_${INCUMBENT_PAIR}_32_p" 1234 "--lr-scale ${lr}" "${tag}"
    done
  done
) &
PID_TARGET_1=$!
wait "${PID_TARGET_0}" "${PID_TARGET_1}"

python scripts/run_v65_eval_sweep.py --pack collision --regimes c1 --pairs "${INCUMBENT_PAIR}" --schedules p --conditions nocache-classon cacheon-classoff --batches 64
python scripts/run_v65_eval_sweep.py --pack delay --regimes d1 --pairs "${INCUMBENT_PAIR}" --schedules p --conditions learned zero random fixed --batches 64
python scripts/build_v65_report.py

INC_CACHE_LR="$(pilot_lr "collision|c1|cacheon-classon|${INCUMBENT_PAIR}")"
INC_NOCACHE_LR="$(pilot_lr "collision|c1|nocache-classon|${INCUMBENT_PAIR}")"
INC_CLASSOFF_LR="$(pilot_lr "collision|c1|cacheon-classoff|${INCUMBENT_PAIR}")"
DELAY_LEARNED_LR="$(pilot_lr "delay|d1|learned|${INCUMBENT_PAIR}")"
DELAY_ZERO_LR="$(pilot_lr "delay|d1|zero|${INCUMBENT_PAIR}")"
DELAY_RANDOM_LR="$(pilot_lr "delay|d1|random|${INCUMBENT_PAIR}")"
DELAY_FIXED_LR="$(pilot_lr "delay|d1|fixed|${INCUMBENT_PAIR}")"

# Delay benchmark validation controls.
run_train 0 "v65_delay_d1_learned_${INCUMBENT_PAIR}_32_p" 1234 "--lr-scale ${DELAY_LEARNED_LR}"
run_train 0 "v65_delay_d1_zero_${INCUMBENT_PAIR}_32_p" 1234 "--lr-scale ${DELAY_ZERO_LR}"
run_train 1 "v65_delay_d1_random_${INCUMBENT_PAIR}_32_p" 1234 "--lr-scale ${DELAY_RANDOM_LR}"
run_train 1 "v65_delay_d1_fixed_${INCUMBENT_PAIR}_32_p" 1234 "--lr-scale ${DELAY_FIXED_LR}"

python scripts/run_v65_eval_sweep.py --pack delay --regimes d1 --pairs "${INCUMBENT_PAIR}" --schedules p --conditions learned zero random fixed --batches 64
python scripts/build_v65_report.py

# Collision Pack main matrix.
(
  for regime in c0 c1 c2; do
    for seed in 2234 3234; do
      run_train 0 "v65_collision_${regime}_cacheon_classon_${INCUMBENT_PAIR}_32_m" "${seed}" "--lr-scale ${INC_CACHE_LR}"
    done
  done
) &
PID_COLLISION_0=$!
(
  for regime in c0 c1 c2; do
    for seed in 2234 3234; do
      run_train 1 "v65_collision_${regime}_nocache_classon_${INCUMBENT_PAIR}_32_m" "${seed}" "--lr-scale ${INC_NOCACHE_LR}"
    done
  done
) &
PID_COLLISION_1=$!
wait "${PID_COLLISION_0}" "${PID_COLLISION_1}"

# Narrow runner-up sensitivity on the hardest collision regime.
run_train 0 "v65_collision_c2_cacheon_classon_${RUNNER_UP_PAIR}_32_m" 4234 "--lr-scale ${RUNNER_CACHE_LR}"
run_train 1 "v65_collision_c2_nocache_classon_${RUNNER_UP_PAIR}_32_m" 4234 "--lr-scale ${RUNNER_CACHE_LR}"

python scripts/run_v65_eval_sweep.py --pack collision --regimes c0 c1 c2 --pairs "${INCUMBENT_PAIR}" "${RUNNER_UP_PAIR}" --schedules m --conditions cacheon-classon nocache-classon --batches 96
python scripts/build_v65_report.py

COLLISION_GATE="$(summary_get 'print(str(bool(obj.get("collision_gate_passed", False))).lower())')"

if [[ "${COLLISION_GATE}" == "true" ]]; then
  (
    for regime in c1 c2; do
      for seed in 2234 3234; do
        run_train 0 "v65_collision_${regime}_cacheon_classoff_${INCUMBENT_PAIR}_32_m" "${seed}" "--lr-scale ${INC_CLASSOFF_LR}"
      done
    done
  )
  python scripts/run_v65_eval_sweep.py --pack collision --regimes c1 c2 --pairs "${INCUMBENT_PAIR}" --schedules m --conditions cacheon-classon cacheon-classoff --batches 96
  python scripts/build_v65_report.py
fi

# Delay Pack main matrix.
(
  for regime in d0 d1 d2; do
    for seed in 2234 3234; do
      run_train 0 "v65_delay_${regime}_learned_${INCUMBENT_PAIR}_32_m" "${seed}" "--lr-scale ${DELAY_LEARNED_LR}"
    done
  done
) &
PID_DELAY_0=$!
(
  for regime in d0 d1 d2; do
    for seed in 2234 3234; do
      run_train 1 "v65_delay_${regime}_zero_${INCUMBENT_PAIR}_32_m" "${seed}" "--lr-scale ${DELAY_ZERO_LR}"
    done
  done
) &
PID_DELAY_1=$!
wait "${PID_DELAY_0}" "${PID_DELAY_1}"

# Runner-up learned-delay sensitivity on the hardest regime.
run_train 0 "v65_delay_d2_learned_${RUNNER_UP_PAIR}_32_m" 2234 "--lr-scale ${RUNNER_CACHE_LR}"
run_train 1 "v65_delay_d2_learned_${RUNNER_UP_PAIR}_32_m" 3234 "--lr-scale ${RUNNER_CACHE_LR}"

python scripts/run_v65_eval_sweep.py --pack delay --regimes d0 d1 d2 --pairs "${INCUMBENT_PAIR}" "${RUNNER_UP_PAIR}" --schedules m --conditions learned zero --batches 96
python scripts/build_v65_report.py

DELAY_GATE="$(summary_get 'print(str(bool(obj.get("delay_gate_passed", False))).lower())')"

if [[ "${DELAY_GATE}" == "true" ]]; then
  run_train 0 "v65_delay_d2_learned_${INCUMBENT_PAIR}_32_l" 5234 "--lr-scale ${DELAY_LEARNED_LR}"
  run_train 0 "v65_delay_d2_adaptive_${INCUMBENT_PAIR}_32_l" 5234 "--lr-scale ${DELAY_LEARNED_LR}"
  run_train 1 "v65_delay_d2_learned_${INCUMBENT_PAIR}_32_l" 6234 "--lr-scale ${DELAY_LEARNED_LR}"
  run_train 1 "v65_delay_d2_adaptive_${INCUMBENT_PAIR}_32_l" 6234 "--lr-scale ${DELAY_LEARNED_LR}"
  python scripts/run_v65_eval_sweep.py --pack delay --regimes d2 --pairs "${INCUMBENT_PAIR}" --schedules l --conditions learned adaptive --batches 128
  python scripts/build_v65_report.py
fi

# Fresh-seed rerun on the strongest promoted collision condition.
if [[ "${COLLISION_GATE}" == "true" ]]; then
  run_train 0 "v65_collision_c2_cacheon_classon_${INCUMBENT_PAIR}_32_l" 5234 "--lr-scale ${INC_CACHE_LR}"
  python scripts/run_v65_eval_sweep.py --pack collision --regimes c2 --pairs "${INCUMBENT_PAIR}" --schedules l --conditions cacheon-classon --batches 128
fi

# Probe collection for source-quality / hard-slice audits.
python scripts/run_v65_eval_sweep.py --pack collision --regimes c2 --pairs "${INCUMBENT_PAIR}" --schedules m --conditions cacheon-classon nocache-classon --batches 96 --collect-probes
if [[ "${COLLISION_GATE}" == "true" ]]; then
  python scripts/run_v65_eval_sweep.py --pack collision --regimes c2 --pairs "${INCUMBENT_PAIR}" --schedules m --conditions cacheon-classoff --batches 96 --collect-probes
fi
python scripts/run_v65_eval_sweep.py --pack delay --regimes d2 --pairs "${INCUMBENT_PAIR}" --schedules m --conditions learned zero --batches 96 --collect-probes

python scripts/build_v65_report.py
pytest -q
bd close "${ISSUE_ID}"
