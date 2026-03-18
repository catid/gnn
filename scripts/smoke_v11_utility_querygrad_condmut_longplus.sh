#!/usr/bin/env bash
set -euo pipefail
SEED="${SEED:-1234}" TRAIN_STEPS_OVERRIDE="${TRAIN_STEPS_OVERRIDE:-220}" RUN_NAME="${RUN_NAME:-v11-utility-querygrad-condmut-smoke}" CONFIG=configs/v11_utility_querygrad_condmut_longplus.yaml DEFAULT_SEED="$SEED" DEFAULT_RUN_NAME="$RUN_NAME" exec "$(dirname "$0")/_train_wrapper.sh"
