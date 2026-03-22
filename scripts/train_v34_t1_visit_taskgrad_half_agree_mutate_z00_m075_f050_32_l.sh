#!/usr/bin/env bash
CONFIG=configs/v34_t1_visit_taskgrad_half_agree_mutate_z00_m075_f050_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v34-t1-visit_taskgrad_half_agree_mutate_z00_m075_f050-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
