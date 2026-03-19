#!/usr/bin/env bash
CONFIG=configs/v22_t2a_querygradonly_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v22-t2a-querygradonly-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
