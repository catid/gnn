#!/usr/bin/env bash
CONFIG=configs/v55_t2b_stageadaptive_late_half_32_xl.yaml DEFAULT_SEED=176234 DEFAULT_RUN_NAME=v55-t2b-stageadaptive_late_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
