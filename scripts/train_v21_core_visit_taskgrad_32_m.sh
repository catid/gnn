#!/usr/bin/env bash
CONFIG=configs/v21_core_visit_taskgrad_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v21-core-visit_taskgrad-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
