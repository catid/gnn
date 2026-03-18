#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v10_utility_querygrad_mutate_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v10-utility-querygrad-mutate-longplus exec "$(dirname "$0")/_train_wrapper.sh"
