#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m apsgnn.train --config configs/v4_retrieval_keycond_search.yaml --run-name v4-retrieval-keycond-smoke --train-steps 250
