#!/usr/bin/env bash
CONFIG=configs/v62_core_visitonly_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v62-core-visitonly-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
