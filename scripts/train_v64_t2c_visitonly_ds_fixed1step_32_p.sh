#!/usr/bin/env bash
CONFIG=configs/v64_t2c_visitonly_ds_fixed1step_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2c-visitonly_ds_fixed1step-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
