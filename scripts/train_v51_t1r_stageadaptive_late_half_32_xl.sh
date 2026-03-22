#!/usr/bin/env bash
CONFIG=configs/v51_t1r_stageadaptive_late_half_32_xl.yaml DEFAULT_SEED=160234 DEFAULT_RUN_NAME=v51-t1r-stageadaptive_late_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
