#!/usr/bin/env bash
CONFIG=configs/v60_t1r_visitonly_32_xl.yaml DEFAULT_SEED=194234 DEFAULT_RUN_NAME=v60-t1r-visitonly-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
