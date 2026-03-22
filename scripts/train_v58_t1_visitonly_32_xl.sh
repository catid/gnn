#!/usr/bin/env bash
CONFIG=configs/v58_t1_visitonly_32_xl.yaml DEFAULT_SEED=190234 DEFAULT_RUN_NAME=v58-t1-visitonly-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
