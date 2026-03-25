#!/usr/bin/env bash
CONFIG=configs/v63ee_core_slowcommit_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-core-slowcommit-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
