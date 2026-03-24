#!/usr/bin/env bash
CONFIG=configs/v63_core_visitonly_dsg_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-core-visitonly_dsg-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
