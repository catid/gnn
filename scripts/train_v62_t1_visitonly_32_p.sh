#!/usr/bin/env bash
CONFIG=configs/v62_t1_visitonly_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v62-t1-visitonly-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
