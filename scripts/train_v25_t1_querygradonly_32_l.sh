#!/usr/bin/env bash
CONFIG=configs/v25_t1_querygradonly_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v25-t1-querygradonly-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
