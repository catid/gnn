#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source .venv/bin/activate

AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [[ "$AVAILABLE_GPUS" -lt 1 ]]; then
  echo "No CUDA devices available for v7 growth mutate hard long run." >&2
  exit 1
fi

REQUESTED_GPUS=4
if [[ "$AVAILABLE_GPUS" -lt "$REQUESTED_GPUS" ]]; then
  NPROC="$AVAILABLE_GPUS"
  echo "Falling back to ${NPROC} GPU(s) for v7 growth mutate hard long training."
else
  NPROC="$REQUESTED_GPUS"
fi

SEED="${SEED:-4234}"
RUN_NAME="${RUN_NAME:-v7-growth-mutate-hard-long-s${SEED}}"

torchrun \
  --standalone \
  --nproc_per_node="$NPROC" \
  -m apsgnn.train \
  --config configs/v7_growth_mutate_hard_long.yaml \
  --run-name "$RUN_NAME" \
  --seed "$SEED"
