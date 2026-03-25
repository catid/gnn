#!/usr/bin/env bash
CONFIG=configs/v64_hmid_visitonly_ds_fixed2step_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-hmid-visitonly_ds_fixed2step-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
