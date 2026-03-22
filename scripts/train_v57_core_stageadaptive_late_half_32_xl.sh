#!/usr/bin/env bash
CONFIG=configs/v57_core_stageadaptive_late_half_32_xl.yaml DEFAULT_SEED=180234 DEFAULT_RUN_NAME=v57-core-stageadaptive_late_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
