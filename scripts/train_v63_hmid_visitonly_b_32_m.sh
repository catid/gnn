#!/usr/bin/env bash
CONFIG=configs/v63_hmid_visitonly_b_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-hmid-visitonly_b-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
