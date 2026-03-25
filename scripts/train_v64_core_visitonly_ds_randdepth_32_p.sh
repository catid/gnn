#!/usr/bin/env bash
CONFIG=configs/v64_core_visitonly_ds_randdepth_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-core-visitonly_ds_randdepth-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
