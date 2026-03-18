#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v18_transfer_t1_visitonly_long.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v18-transfer-t1-visitonly-long exec "$(dirname "$0")/_smoke_wrapper.sh"
