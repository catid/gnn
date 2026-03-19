#!/usr/bin/env bash
CONFIG=configs/v21_core_querygradonly_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v21-core-querygradonly-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
