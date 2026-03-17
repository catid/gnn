#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate
python -m apsgnn.train --config configs/smoke.yaml --run-name smoke
