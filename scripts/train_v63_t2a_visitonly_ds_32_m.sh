#!/usr/bin/env bash
CONFIG=configs/v63_t2a_visitonly_ds_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t2a-visitonly_ds-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
