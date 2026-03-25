#!/usr/bin/env bash
CONFIG=configs/v63ee_t2a_gonline_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t2a-gonline-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
