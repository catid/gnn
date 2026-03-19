#!/usr/bin/env bash
CONFIG=configs/v21_t1_full_querygrad_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v21-t1-full_querygrad-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
