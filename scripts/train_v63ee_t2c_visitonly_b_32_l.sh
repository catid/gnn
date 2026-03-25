#!/usr/bin/env bash
CONFIG=configs/v63ee_t2c_visitonly_b_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t2c-visitonly_b-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
