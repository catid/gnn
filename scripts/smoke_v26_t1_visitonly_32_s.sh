#!/usr/bin/env bash
CONFIG=configs/v26_t1_visitonly_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-t1-visitonly-32-s exec "$(dirname "$0")/_smoke_wrapper_single_gpu.sh"
