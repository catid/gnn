#!/usr/bin/env bash
CONFIG=configs/v58_core_visitonly_32_xl.yaml DEFAULT_SEED=190234 DEFAULT_RUN_NAME=v58-core-visitonly-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
