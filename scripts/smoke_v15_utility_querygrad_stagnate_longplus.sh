#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v15_utility_querygrad_stagnate_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v15-utility-querygrad-stagnate-smoke TRAIN_STEPS_OVERRIDE=220 exec "$(dirname "$0")/_train_wrapper.sh"
