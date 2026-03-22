#!/usr/bin/env bash
CONFIG=configs/v54_t2b_stageadaptive_late_half_32_xl.yaml DEFAULT_SEED=174234 DEFAULT_RUN_NAME=v54-t2b-stageadaptive_late_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
