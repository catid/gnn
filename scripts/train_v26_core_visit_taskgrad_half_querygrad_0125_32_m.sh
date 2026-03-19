#!/usr/bin/env bash
CONFIG=configs/v26_core_visit_taskgrad_half_querygrad_0125_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-core-visit_taskgrad_half_querygrad_0125-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
