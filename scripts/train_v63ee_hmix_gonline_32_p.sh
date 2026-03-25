#!/usr/bin/env bash
CONFIG=configs/v63ee_hmix_gonline_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-hmix-gonline-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
