#!/usr/bin/env bash
CONFIG=configs/v52_t1_stageadaptive_late_half_32_xl.yaml DEFAULT_SEED=170234 DEFAULT_RUN_NAME=v52-t1-stageadaptive_late_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
