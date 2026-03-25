#!/usr/bin/env bash
CONFIG=configs/v63ee_hmix_visitonly_d_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-hmix-visitonly_d-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
