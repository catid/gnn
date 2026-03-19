#!/usr/bin/env bash
CONFIG=configs/v26_t1_visit_taskgrad_0625_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-t1-visit_taskgrad_0625-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
