#!/usr/bin/env bash
CONFIG=configs/v63ee_hmix_stage_early_vt_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-hmix-stage_early_vt-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
