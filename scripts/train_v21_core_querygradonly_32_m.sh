#!/usr/bin/env bash
CONFIG=configs/v21_core_querygradonly_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v21-core-querygradonly-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
