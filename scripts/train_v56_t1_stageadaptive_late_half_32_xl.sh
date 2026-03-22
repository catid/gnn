#!/usr/bin/env bash
CONFIG=configs/v56_t1_stageadaptive_late_half_32_xl.yaml DEFAULT_SEED=178234 DEFAULT_RUN_NAME=v56-t1-stageadaptive_late_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
