#!/usr/bin/env bash
CONFIG=configs/v61_t1_visit_taskgrad_half_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-t1-visit_taskgrad_half-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
