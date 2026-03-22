#!/usr/bin/env bash
CONFIG=configs/v47_t1_visit_taskgrad_half_32_xl.yaml DEFAULT_SEED=120234 DEFAULT_RUN_NAME=v47-t1-visit_taskgrad_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
