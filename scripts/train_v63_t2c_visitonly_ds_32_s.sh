#!/usr/bin/env bash
CONFIG=configs/v63_t2c_visitonly_ds_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t2c-visitonly_ds-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
