#!/usr/bin/env bash
CONFIG=configs/v64_t2a_visitonly_ds_fixed2step_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2a-visitonly_ds_fixed2step-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
