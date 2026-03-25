#!/usr/bin/env bash
CONFIG=configs/v64_t2c_visitonly_ds_fixed2step_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2c-visitonly_ds_fixed2step-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
