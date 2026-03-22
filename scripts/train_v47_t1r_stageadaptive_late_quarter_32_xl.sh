#!/usr/bin/env bash
CONFIG=configs/v47_t1r_stageadaptive_late_quarter_32_xl.yaml DEFAULT_SEED=120234 DEFAULT_RUN_NAME=v47-t1r-stageadaptive_late_quarter-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
