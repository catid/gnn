#!/usr/bin/env bash
CONFIG=configs/v27_t1_visitonly_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v27-t1-visitonly-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
