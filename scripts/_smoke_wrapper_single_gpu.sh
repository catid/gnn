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
train_steps=${TRAIN_STEPS_OVERRIDE:-220}

CUDA_VISIBLE_DEVICES="$gpu_id" python -m apsgnn.train \
  --config "$CONFIG" \
  --run-name "$run_name" \
  --seed "$seed" \
  --train-steps "$train_steps"
