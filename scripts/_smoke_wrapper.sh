#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

available=$(python -c "import torch; print(torch.cuda.device_count())")
requested=4
if [[ "$available" -lt "$requested" ]]; then
  echo "Requested ${requested} GPUs but found ${available}; falling back." >&2
fi
nproc=$(( available < requested ? available : requested ))
if [[ "$nproc" -lt 1 ]]; then
  echo "No CUDA devices available." >&2
  exit 1
fi

if [[ -z "${CONFIG:-}" || -z "${DEFAULT_SEED:-}" || -z "${DEFAULT_RUN_NAME:-}" ]]; then
  echo "CONFIG, DEFAULT_SEED, and DEFAULT_RUN_NAME must be set." >&2
  exit 1
fi

seed=${SEED:-$DEFAULT_SEED}
run_name=${RUN_NAME:-${DEFAULT_RUN_NAME}-s${seed}}
train_steps=${TRAIN_STEPS_OVERRIDE:-220}

torchrun --standalone --nproc_per_node="$nproc" -m apsgnn.train \
  --config "$CONFIG" \
  --run-name "$run_name" \
  --seed "$seed" \
  --train-steps "$train_steps"
