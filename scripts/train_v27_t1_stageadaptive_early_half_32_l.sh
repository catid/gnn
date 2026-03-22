#!/usr/bin/env bash
CONFIG=configs/v27_t1_stageadaptive_early_half_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v27-t1-stageadaptive_early_half-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
