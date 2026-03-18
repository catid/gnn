#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v17_transfer_h1_utility_querygradonly_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v17-transfer-h1-utility-querygradonly-longplus exec "$(dirname "$0")/_train_wrapper.sh"
