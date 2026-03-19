#!/usr/bin/env bash
CONFIG=configs/v25_t1_visitonly_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v25-t1-visitonly-32-l exec "$(dirname "$0")/_smoke_wrapper_single_gpu.sh"
