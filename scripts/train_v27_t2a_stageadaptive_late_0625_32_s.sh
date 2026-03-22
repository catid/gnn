#!/usr/bin/env bash
CONFIG=configs/v27_t2a_stageadaptive_late_0625_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v27-t2a-stageadaptive_late_0625-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
