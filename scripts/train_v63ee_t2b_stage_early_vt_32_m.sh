#!/usr/bin/env bash
CONFIG=configs/v63ee_t2b_stage_early_vt_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t2b-stage_early_vt-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
