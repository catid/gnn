#!/usr/bin/env bash
CONFIG=configs/v39_t1r_visit_taskgrad_half_agree_mutate_z00_m075_f025_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v39-t1r-visit_taskgrad_half_agree_mutate_z00_m075_f025-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
