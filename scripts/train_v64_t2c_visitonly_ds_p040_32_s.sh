#!/usr/bin/env bash
CONFIG=configs/v64_t2c_visitonly_ds_p040_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2c-visitonly_ds_p040-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
