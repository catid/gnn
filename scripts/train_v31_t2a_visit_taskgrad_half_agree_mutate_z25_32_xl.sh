#!/usr/bin/env bash
CONFIG=configs/v31_t2a_visit_taskgrad_half_agree_mutate_z25_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v31-t2a-visit_taskgrad_half_agree_mutate_z25-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
