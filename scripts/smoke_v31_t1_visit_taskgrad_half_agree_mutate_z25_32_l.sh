#!/usr/bin/env bash
CONFIG=configs/v31_t1_visit_taskgrad_half_agree_mutate_z25_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v31-t1-visit_taskgrad_half_agree_mutate_z25-32-l exec "$(dirname "$0")/_smoke_wrapper_single_gpu.sh"
