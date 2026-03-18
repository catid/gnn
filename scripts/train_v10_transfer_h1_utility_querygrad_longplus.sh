#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v10_transfer_h1_utility_querygrad_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v10-transfer-h1-utility-querygrad-longplus exec "$(dirname "$0")/_train_wrapper.sh"
