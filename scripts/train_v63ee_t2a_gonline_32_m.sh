#!/usr/bin/env bash
CONFIG=configs/v63ee_t2a_gonline_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t2a-gonline-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
