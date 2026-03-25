#!/usr/bin/env bash
CONFIG=configs/v64_t1r_visitonly_ds_auxanneal_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t1r-visitonly_ds_auxanneal-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
