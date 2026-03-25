#!/usr/bin/env bash
CONFIG=configs/v64_hmix_visitonly_ds_auxanneal_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-hmix-visitonly_ds_auxanneal-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
