#!/usr/bin/env bash
CONFIG=configs/v45_t1r_stageadaptive_late_half_agree_mutate_z00_m075_f0125_32_xl.yaml DEFAULT_SEED=100234 DEFAULT_RUN_NAME=v45-t1r-stageadaptive_late_half_agree_mutate_z00_m075_f0125-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
