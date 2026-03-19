#!/usr/bin/env bash
CONFIG=configs/v26_core_visit_taskgrad_025_32_p_lr100.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-core-visit_taskgrad_025-32-p-lr100 exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
