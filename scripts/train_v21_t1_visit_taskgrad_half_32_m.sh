#!/usr/bin/env bash
CONFIG=configs/v21_t1_visit_taskgrad_half_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v21-t1-visit_taskgrad_half-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
