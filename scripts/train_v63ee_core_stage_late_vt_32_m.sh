#!/usr/bin/env bash
CONFIG=configs/v63ee_core_stage_late_vt_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-core-stage_late_vt-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
