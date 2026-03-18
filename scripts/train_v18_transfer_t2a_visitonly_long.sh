#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v18_transfer_t2a_visitonly_long.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v18-transfer-t2a-visitonly-long exec "$(dirname "$0")/_train_wrapper.sh"
