#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
export PATH="$HOME/.local/bin:$PATH"

ISSUE_ID="${1:-gnn-856}"
SUMMARY="reports/summary_metrics_v66.json"

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
obj = json.loads(Path("reports/summary_metrics_v66.json").read_text())
choice = obj.get("pilot_choices", {}).get(family, {})
print(choice.get("lr_multiplier", 0.8))
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

python scripts/gen_v66_configs.py
python -m compileall apsgnn apsgnn/probes.py scripts/gen_v66_configs.py scripts/run_v66_eval_sweep.py scripts/build_v66_report.py tests/test_v66_forensic_headroom.py
pytest -q tests/test_v66_forensic_headroom.py
pytest -q

# 100-step calibration profile on C1.
GPU_ID=0 RUN_NAME="v66-collision-c1-cacheon-visit_taskgrad_half_d-32-p-profile-s1234" \
TRAIN_STEPS_OVERRIDE=100 TRAIN_LOG_TO_FILE=1 \
TRAIN_LOG_PATH="runs/v66-collision-c1-cacheon-visit_taskgrad_half_d-32-p-profile-s1234.console.log" \
bash scripts/train_v66_collision_c1_cacheon_visit_taskgrad_half_d_32_p.sh

# Fair-shot pilots: collision intervention families and delay families.
(
  for condition in cacheon nocache recent1 topk1; do
    for lr in 0.6 0.8 1.0; do
      tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
      run_train 0 "v66_collision_c2_${condition}_visit_taskgrad_half_d_32_p" 1234 "--lr-scale ${lr}" "${tag}"
    done
  done
) &
PID_PILOT_0=$!
(
  for condition in cacheon nocache; do
    for lr in 0.6 0.8 1.0; do
      tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
      run_train 1 "v66_collision_c2_${condition}_visitonly_d_32_p" 1234 "--lr-scale ${lr}" "${tag}"
    done
  done
  for regime in d2 rd2; do
    for condition in learned zero; do
      for lr in 0.6 0.8 1.0; do
        tag="lr$(printf '%02d' "$(python3 - <<PY
lr=${lr}
print(int(round(lr * 10)))
PY
)")"
        run_train 1 "v66_delay_${regime}_${condition}_visit_taskgrad_half_d_32_p" 1234 "--lr-scale ${lr}" "${tag}"
      done
    done
  done
) &
PID_PILOT_1=$!
wait "${PID_PILOT_0}" "${PID_PILOT_1}"

python scripts/run_v66_eval_sweep.py --pack collision --regimes c2 --pairs visit_taskgrad_half_d visitonly_d --schedules p --conditions cacheon nocache recent1 topk1 --batches 64
python scripts/run_v66_eval_sweep.py --pack delay --regimes d2 rd2 --pairs visit_taskgrad_half_d --schedules p --conditions learned zero --batches 64
python scripts/build_v66_report.py

INC_CACHE_LR="$(pilot_lr "collision|c2|cacheon|visit_taskgrad_half_d")"
INC_NOCACHE_LR="$(pilot_lr "collision|c2|nocache|visit_taskgrad_half_d")"
INC_RECENT1_LR="$(pilot_lr "collision|c2|recent1|visit_taskgrad_half_d")"
INC_TOPK1_LR="$(pilot_lr "collision|c2|topk1|visit_taskgrad_half_d")"
CTRL_CACHE_LR="$(pilot_lr "collision|c2|cacheon|visitonly_d")"
CTRL_NOCACHE_LR="$(pilot_lr "collision|c2|nocache|visitonly_d")"
OLD_D2_LEARNED_LR="$(pilot_lr "delay|d2|learned|visit_taskgrad_half_d")"
OLD_D2_ZERO_LR="$(pilot_lr "delay|d2|zero|visit_taskgrad_half_d")"
RD2_LEARNED_LR="$(pilot_lr "delay|rd2|learned|visit_taskgrad_half_d")"
RD2_ZERO_LR="$(pilot_lr "delay|rd2|zero|visit_taskgrad_half_d")"

# Validation controls for the delay audit families.
run_train 0 "v66_delay_d2_random_visit_taskgrad_half_d_32_p" 1234 "--lr-scale ${OLD_D2_LEARNED_LR}"
run_train 0 "v66_delay_d2_fixed_visit_taskgrad_half_d_32_p" 1234 "--lr-scale ${OLD_D2_LEARNED_LR}"
run_train 1 "v66_delay_rd2_random_visit_taskgrad_half_d_32_p" 1234 "--lr-scale ${RD2_LEARNED_LR}"
run_train 1 "v66_delay_rd2_fixed_visit_taskgrad_half_d_32_p" 1234 "--lr-scale ${RD2_LEARNED_LR}"
run_train 1 "v66_delay_rd2_required_visit_taskgrad_half_d_32_p" 1234 "--lr-scale ${RD2_LEARNED_LR}"

python scripts/run_v66_eval_sweep.py --pack delay --regimes d2 rd2 --pairs visit_taskgrad_half_d --schedules p --conditions learned zero random fixed required --batches 64
python scripts/build_v66_report.py

# Bundle A: collision forensic matrix, first seed.
(
  run_train 0 "v66_collision_c1_cacheon_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${INC_CACHE_LR}"
  run_train 0 "v66_collision_c2_cacheon_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${INC_CACHE_LR}"
  run_train 0 "v66_collision_c1_recent1_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${INC_RECENT1_LR}"
  run_train 0 "v66_collision_c2_recent1_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${INC_RECENT1_LR}"
) &
PID_COLLISION_0=$!
(
  run_train 1 "v66_collision_c1_nocache_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${INC_NOCACHE_LR}"
  run_train 1 "v66_collision_c2_nocache_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${INC_NOCACHE_LR}"
  run_train 1 "v66_collision_c1_topk1_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${INC_TOPK1_LR}"
  run_train 1 "v66_collision_c2_topk1_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${INC_TOPK1_LR}"
) &
PID_COLLISION_1=$!
wait "${PID_COLLISION_0}" "${PID_COLLISION_1}"

python scripts/run_v66_eval_sweep.py --pack collision --regimes c1 c2 --pairs visit_taskgrad_half_d --schedules m --conditions cacheon nocache recent1 topk1 --batches 96
python scripts/build_v66_report.py

BEST_COLLISION_INTERVENTION="$(summary_get 'print(obj.get("collision_pack", {}).get("best_intervention", "recent1"))')"
if [[ -z "${BEST_COLLISION_INTERVENTION}" ]]; then
  BEST_COLLISION_INTERVENTION="recent1"
fi
BEST_COLLISION_LR="${INC_RECENT1_LR}"
if [[ "${BEST_COLLISION_INTERVENTION}" == "topk1" ]]; then
  BEST_COLLISION_LR="${INC_TOPK1_LR}"
fi

# Bundle A: verification rerun for the strongest intervention and narrow runner-up control.
(
  run_train 0 "v66_collision_c1_${BEST_COLLISION_INTERVENTION}_visit_taskgrad_half_d_32_m" 3234 "--lr-scale ${BEST_COLLISION_LR}"
  run_train 1 "v66_collision_c2_${BEST_COLLISION_INTERVENTION}_visit_taskgrad_half_d_32_m" 3234 "--lr-scale ${BEST_COLLISION_LR}"
) &
PID_COLLISION_VERIFY=$!
(
  run_train 0 "v66_collision_c2_cacheon_visitonly_d_32_m" 4234 "--lr-scale ${CTRL_CACHE_LR}"
  run_train 1 "v66_collision_c2_nocache_visitonly_d_32_m" 4234 "--lr-scale ${CTRL_NOCACHE_LR}"
) &
PID_COLLISION_CTRL=$!
wait "${PID_COLLISION_VERIFY}" "${PID_COLLISION_CTRL}"

python scripts/run_v66_eval_sweep.py --pack collision --regimes c1 c2 --pairs visit_taskgrad_half_d visitonly_d --schedules m --conditions cacheon nocache recent1 topk1 --batches 96
python scripts/run_v66_eval_sweep.py --pack collision --regimes c2 --pairs visit_taskgrad_half_d --schedules m --conditions cacheon nocache "${BEST_COLLISION_INTERVENTION}" --batches 96 --collect-probes
python scripts/run_v66_eval_sweep.py --pack collision --regimes c1 c2 --pairs visit_taskgrad_half_d --schedules m --conditions cacheon "${BEST_COLLISION_INTERVENTION}" --batches 96 --override-cache-bypass-mode zero_update
python scripts/build_v66_report.py

COLLISION_POSITIVE="$(summary_get 'print(str(bool(obj.get("collision_pack", {}).get("positive", False))).lower())')"
BEST_COLLISION_RECOVERY="$(summary_get 'print(obj.get("collision_pack", {}).get("best_intervention_mean_recovery_fraction", 0.0))')"
if [[ "${COLLISION_POSITIVE}" == "true" ]] && python3 - <<PY
value = float("${BEST_COLLISION_RECOVERY}")
raise SystemExit(0 if value > 0.15 else 1)
PY
then
  run_train 0 "v66_collision_c2_${BEST_COLLISION_INTERVENTION}_visit_taskgrad_half_d_32_l" 5234 "--lr-scale ${BEST_COLLISION_LR}"
  python scripts/run_v66_eval_sweep.py --pack collision --regimes c2 --pairs visit_taskgrad_half_d --schedules l --conditions "${BEST_COLLISION_INTERVENTION}" --batches 128
  python scripts/build_v66_report.py
fi

# Bundle B: current delay audit reruns plus redesigned benchmark matrix.
(
  run_train 0 "v66_delay_d1_learned_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${OLD_D2_LEARNED_LR}"
  run_train 0 "v66_delay_d2_learned_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${OLD_D2_LEARNED_LR}"
  run_train 0 "v66_delay_rd1_learned_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${RD2_LEARNED_LR}"
  run_train 0 "v66_delay_rd1_learned_visit_taskgrad_half_d_32_m" 3234 "--lr-scale ${RD2_LEARNED_LR}"
  run_train 0 "v66_delay_rd2_learned_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${RD2_LEARNED_LR}"
  run_train 0 "v66_delay_rd2_learned_visit_taskgrad_half_d_32_m" 3234 "--lr-scale ${RD2_LEARNED_LR}"
) &
PID_DELAY_0=$!
(
  run_train 1 "v66_delay_d1_zero_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${OLD_D2_ZERO_LR}"
  run_train 1 "v66_delay_d2_zero_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${OLD_D2_ZERO_LR}"
  run_train 1 "v66_delay_rd1_zero_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${RD2_ZERO_LR}"
  run_train 1 "v66_delay_rd1_zero_visit_taskgrad_half_d_32_m" 3234 "--lr-scale ${RD2_ZERO_LR}"
  run_train 1 "v66_delay_rd2_zero_visit_taskgrad_half_d_32_m" 2234 "--lr-scale ${RD2_ZERO_LR}"
  run_train 1 "v66_delay_rd2_zero_visit_taskgrad_half_d_32_m" 3234 "--lr-scale ${RD2_ZERO_LR}"
) &
PID_DELAY_1=$!
wait "${PID_DELAY_0}" "${PID_DELAY_1}"

python scripts/run_v66_eval_sweep.py --pack delay --regimes d1 d2 rd1 rd2 --pairs visit_taskgrad_half_d --schedules m --conditions learned zero --batches 96
python scripts/run_v66_eval_sweep.py --pack delay --regimes d2 rd2 --pairs visit_taskgrad_half_d --schedules m --conditions learned zero --batches 96 --collect-probes
python scripts/build_v66_report.py

DELAY_POSITIVE="$(summary_get 'print(str(bool(obj.get("delay_pack", {}).get("redesign_positive", False))).lower())')"
if [[ "${DELAY_POSITIVE}" == "true" ]]; then
  run_train 0 "v66_delay_rd2_learned_visit_taskgrad_half_d_32_l" 5234 "--lr-scale ${RD2_LEARNED_LR}"
  run_train 1 "v66_delay_rd2_zero_visit_taskgrad_half_d_32_l" 5234 "--lr-scale ${RD2_ZERO_LR}"
  python scripts/run_v66_eval_sweep.py --pack delay --regimes rd2 --pairs visit_taskgrad_half_d --schedules l --conditions learned zero --batches 128
  python scripts/run_v66_eval_sweep.py --pack delay --regimes rd2 --pairs visit_taskgrad_half_d --schedules l --conditions learned zero --batches 128 --collect-probes
  python scripts/build_v66_report.py
fi

pytest -q tests/test_v66_forensic_headroom.py
pytest -q

bd update "${ISSUE_ID}" --append-notes "Completed v66 forensic headroom campaign, rebuilt reports, and reran full verification."
bd close "${ISSUE_ID}"
