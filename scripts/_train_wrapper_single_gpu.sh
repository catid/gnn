#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

if [[ -z "${CONFIG:-}" || -z "${DEFAULT_SEED:-}" || -z "${DEFAULT_RUN_NAME:-}" ]]; then
  echo "CONFIG, DEFAULT_SEED, and DEFAULT_RUN_NAME must be set." >&2
  exit 1
fi

gpu_id=${GPU_ID:-0}
seed=${SEED:-$DEFAULT_SEED}
run_name=${RUN_NAME:-${DEFAULT_RUN_NAME}-s${seed}}
log_to_file=${TRAIN_LOG_TO_FILE:-0}
log_path=${TRAIN_LOG_PATH:-runs/${run_name}.console.log}

cmd=(
  python -m apsgnn.train
  --config "$CONFIG"
  --run-name "$run_name"
  --seed "$seed"
)
if [[ -n "${TRAIN_STEPS_OVERRIDE:-}" ]]; then
  cmd+=(--train-steps "$TRAIN_STEPS_OVERRIDE")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS:-}"
  cmd+=("${extra_args[@]}")
fi

if [[ "$log_to_file" == "1" ]]; then
  mkdir -p "$(dirname "$log_path")"
  CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}" >"$log_path" 2>&1
else
  CUDA_VISIBLE_DEVICES="$gpu_id" "${cmd[@]}"
fi
