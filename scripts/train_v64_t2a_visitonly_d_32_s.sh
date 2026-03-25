#!/usr/bin/env bash
CONFIG=configs/v64_t2a_visitonly_d_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2a-visitonly_d-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
