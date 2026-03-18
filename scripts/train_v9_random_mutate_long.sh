#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v9_random_mutate_long.yaml DEFAULT_SEED=4234 DEFAULT_RUN_NAME=v9-random-mutate-long exec "$(dirname "$0")/_train_wrapper.sh"
