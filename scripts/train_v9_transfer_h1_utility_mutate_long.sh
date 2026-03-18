#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v9_transfer_h1_utility_mutate_long.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v9-transfer-h1-utility-mutate-long exec "$(dirname "$0")/_train_wrapper.sh"
