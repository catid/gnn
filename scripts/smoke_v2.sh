#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m apsgnn.train --config configs/v2_smoke.yaml --run-name v2-smoke
