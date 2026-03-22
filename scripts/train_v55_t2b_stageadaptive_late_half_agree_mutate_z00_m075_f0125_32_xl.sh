#!/usr/bin/env bash
CONFIG=configs/v55_t2b_stageadaptive_late_half_agree_mutate_z00_m075_f0125_32_xl.yaml DEFAULT_SEED=176234 DEFAULT_RUN_NAME=v55-t2b-stageadaptive_late_half_agree_mutate_z00_m075_f0125-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
