#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v9_utility_selective_long.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v9-utility-selective-long exec "$(dirname "$0")/_train_wrapper.sh"
