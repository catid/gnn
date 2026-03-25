#!/usr/bin/env bash
CONFIG=configs/v63ee_hmix_slowcommit_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-hmix-slowcommit-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
