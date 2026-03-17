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

seed=${SEED:-1234}
run_name=${RUN_NAME:-v7-growth-mutate-hard-s${seed}}
cmd=(
  torchrun --standalone --nproc_per_node="$nproc" -m apsgnn.train
  --config configs/v7_growth_mutate_hard.yaml
  --run-name "$run_name"
  --seed "$seed"
)
if [[ -n "${TRAIN_STEPS_OVERRIDE:-}" ]]; then
  cmd+=(--train-steps "$TRAIN_STEPS_OVERRIDE")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  cmd+=("${extra_args[@]}")
fi
"${cmd[@]}"
