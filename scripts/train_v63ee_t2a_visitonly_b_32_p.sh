#!/usr/bin/env bash
CONFIG=configs/v63ee_t2a_visitonly_b_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t2a-visitonly_b-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
