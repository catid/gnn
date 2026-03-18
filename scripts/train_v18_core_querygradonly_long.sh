#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v18_core_querygradonly_long.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v18-core-querygradonly-long exec "$(dirname "$0")/_train_wrapper.sh"
