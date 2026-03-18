#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v10_transfer_h1_utility_querymix_mutate_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v10-transfer-h1-utility-querymix-mutate-longplus exec "$(dirname "$0")/_train_wrapper.sh"
