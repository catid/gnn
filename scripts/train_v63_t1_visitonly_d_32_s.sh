#!/usr/bin/env bash
CONFIG=configs/v63_t1_visitonly_d_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t1-visitonly_d-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
