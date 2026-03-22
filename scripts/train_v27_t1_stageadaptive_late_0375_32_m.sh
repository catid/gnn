#!/usr/bin/env bash
CONFIG=configs/v27_t1_stageadaptive_late_0375_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v27-t1-stageadaptive_late_0375-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
