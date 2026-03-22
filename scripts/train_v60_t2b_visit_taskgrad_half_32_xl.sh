#!/usr/bin/env bash
CONFIG=configs/v60_t2b_visit_taskgrad_half_32_xl.yaml DEFAULT_SEED=194234 DEFAULT_RUN_NAME=v60-t2b-visit_taskgrad_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
