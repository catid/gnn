#!/usr/bin/env bash
CONFIG=configs/v63ee_t1_stage_early_vt_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t1-stage_early_vt-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
