#!/usr/bin/env bash
CONFIG=configs/v35_core_visit_taskgrad_half_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v35-core-visit_taskgrad_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
