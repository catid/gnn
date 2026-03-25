#!/usr/bin/env bash
CONFIG=configs/v63ee_hmix_visitonly_b_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-hmix-visitonly_b-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
