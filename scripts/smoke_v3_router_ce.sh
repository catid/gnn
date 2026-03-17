#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m apsgnn.train --config configs/v3_router_ce_search.yaml --run-name v3-router-ce-smoke --train-steps 250
