#!/usr/bin/env bash
CONFIG=configs/v49_t1_stageadaptive_final_half_32_xl.yaml DEFAULT_SEED=140234 DEFAULT_RUN_NAME=v49-t1-stageadaptive_final_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
