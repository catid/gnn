#!/usr/bin/env bash
CONFIG=configs/v44_t1_stageadaptive_late_half_agree_mutate_z00_m075_f025_32_xl.yaml DEFAULT_SEED=90234 DEFAULT_RUN_NAME=v44-t1-stageadaptive_late_half_agree_mutate_z00_m075_f025-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
