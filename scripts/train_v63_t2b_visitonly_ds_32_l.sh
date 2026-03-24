#!/usr/bin/env bash
CONFIG=configs/v63_t2b_visitonly_ds_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t2b-visitonly_ds-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
