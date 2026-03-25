#!/usr/bin/env bash
CONFIG=configs/v64_t1_visitonly_ds_core_best_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t1-visitonly_ds_core_best-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
