#!/usr/bin/env bash
CONFIG=configs/v63ee_core_slowcommit_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-core-slowcommit-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
