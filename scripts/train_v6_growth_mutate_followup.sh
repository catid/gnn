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

torchrun --standalone --nproc_per_node="$nproc" -m apsgnn.train --config configs/v6_growth_mutate_followup.yaml --run-name v6-growth-mutate-hard
