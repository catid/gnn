#!/usr/bin/env bash
CONFIG=configs/v63ee_hmix_stable_vt_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-hmix-stable_vt-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
