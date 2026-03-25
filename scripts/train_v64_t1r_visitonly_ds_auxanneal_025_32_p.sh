#!/usr/bin/env bash
CONFIG=configs/v64_t1r_visitonly_ds_auxanneal_025_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t1r-visitonly_ds_auxanneal_025-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
