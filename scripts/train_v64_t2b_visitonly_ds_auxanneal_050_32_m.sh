#!/usr/bin/env bash
CONFIG=configs/v64_t2b_visitonly_ds_auxanneal_050_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2b-visitonly_ds_auxanneal_050-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
