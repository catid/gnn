#!/usr/bin/env bash
CONFIG=configs/v64_t1r_visitonly_ds_p005_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t1r-visitonly_ds_p005-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
