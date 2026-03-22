#!/usr/bin/env bash
CONFIG=configs/v61_hmid_visitonly_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-hmid-visitonly-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
