#!/usr/bin/env bash
CONFIG=configs/v49_t2a_stageadaptive_late_half_32_xl.yaml DEFAULT_SEED=140234 DEFAULT_RUN_NAME=v49-t2a-stageadaptive_late_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
