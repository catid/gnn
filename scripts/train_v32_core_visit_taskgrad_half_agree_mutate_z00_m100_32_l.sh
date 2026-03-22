#!/usr/bin/env bash
CONFIG=configs/v32_core_visit_taskgrad_half_agree_mutate_z00_m100_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v32-core-visit_taskgrad_half_agree_mutate_z00_m100-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
