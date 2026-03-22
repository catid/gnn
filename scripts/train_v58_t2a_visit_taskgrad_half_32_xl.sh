#!/usr/bin/env bash
CONFIG=configs/v58_t2a_visit_taskgrad_half_32_xl.yaml DEFAULT_SEED=190234 DEFAULT_RUN_NAME=v58-t2a-visit_taskgrad_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
