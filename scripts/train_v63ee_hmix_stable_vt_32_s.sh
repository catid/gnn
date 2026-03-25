#!/usr/bin/env bash
CONFIG=configs/v63ee_hmix_stable_vt_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-hmix-stable_vt-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
