#!/usr/bin/env bash
CONFIG=configs/v32_core_visit_taskgrad_half_agree_mutate_z00_m050_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v32-core-visit_taskgrad_half_agree_mutate_z00_m050-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
