#!/usr/bin/env bash
CONFIG=configs/v64_t2a_visitonly_ds_p025_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2a-visitonly_ds_p025-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
