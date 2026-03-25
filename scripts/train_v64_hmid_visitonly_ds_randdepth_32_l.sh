#!/usr/bin/env bash
CONFIG=configs/v64_hmid_visitonly_ds_randdepth_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-hmid-visitonly_ds_randdepth-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
