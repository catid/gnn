#!/usr/bin/env bash
CONFIG=configs/v63_hmix_visitonly_d_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-hmix-visitonly_d-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
