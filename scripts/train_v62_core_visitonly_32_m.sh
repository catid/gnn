#!/usr/bin/env bash
CONFIG=configs/v62_core_visitonly_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v62-core-visitonly-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
