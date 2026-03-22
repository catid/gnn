#!/usr/bin/env bash
CONFIG=configs/v42_t1_stageadaptive_late_half_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v42-t1-stageadaptive_late_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
