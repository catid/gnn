#!/usr/bin/env bash
CONFIG=configs/v63ee_t2c_stable_v_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t2c-stable_v-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
