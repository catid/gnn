#!/usr/bin/env bash
CONFIG=configs/v26_t1_visitonly_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-t1-visitonly-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
