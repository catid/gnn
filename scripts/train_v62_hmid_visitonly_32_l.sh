#!/usr/bin/env bash
CONFIG=configs/v62_hmid_visitonly_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v62-hmid-visitonly-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
