#!/usr/bin/env bash
CONFIG=configs/v59_t1_visitonly_32_xl.yaml DEFAULT_SEED=192234 DEFAULT_RUN_NAME=v59-t1-visitonly-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
