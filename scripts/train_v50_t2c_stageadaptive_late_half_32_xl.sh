#!/usr/bin/env bash
CONFIG=configs/v50_t2c_stageadaptive_late_half_32_xl.yaml DEFAULT_SEED=150234 DEFAULT_RUN_NAME=v50-t2c-stageadaptive_late_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
