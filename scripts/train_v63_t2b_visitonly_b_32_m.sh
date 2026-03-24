#!/usr/bin/env bash
CONFIG=configs/v63_t2b_visitonly_b_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t2b-visitonly_b-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
