#!/usr/bin/env bash
CONFIG=configs/v21_t2a_full_querygrad_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v21-t2a-full_querygrad-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
