#!/usr/bin/env bash
CONFIG=configs/v62_hmix_visitonly_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v62-hmix-visitonly-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
