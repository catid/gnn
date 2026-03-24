#!/usr/bin/env bash
CONFIG=configs/v63_t2c_visitonly_dsg_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t2c-visitonly_dsg-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
