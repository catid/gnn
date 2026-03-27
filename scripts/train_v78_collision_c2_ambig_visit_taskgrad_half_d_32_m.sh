#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1234}"
RUN_NAME="${RUN_NAME:-}"
TRAIN_STEPS_OVERRIDE="${TRAIN_STEPS_OVERRIDE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
ARGS=(--config "/home/catid/gnn/configs/v78_collision_c2_ambig_visit_taskgrad_half_d_32_m.yaml" --seed "${SEED}")
if [[ -n "${RUN_NAME}" ]]; then ARGS+=(--run-name "${RUN_NAME}"); fi
if [[ -n "${TRAIN_STEPS_OVERRIDE}" ]]; then ARGS+=(--train-steps "${TRAIN_STEPS_OVERRIDE}"); fi
if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  ARGS+=("${EXTRA_ARR[@]}")
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
python3 -m apsgnn.train "${ARGS[@]}"
