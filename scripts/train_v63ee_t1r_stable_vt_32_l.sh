#!/usr/bin/env bash
CONFIG=configs/v63ee_t1r_stable_vt_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t1r-stable_vt-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
