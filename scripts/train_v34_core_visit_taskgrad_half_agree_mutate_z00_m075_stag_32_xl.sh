#!/usr/bin/env bash
CONFIG=configs/v34_core_visit_taskgrad_half_agree_mutate_z00_m075_stag_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v34-core-visit_taskgrad_half_agree_mutate_z00_m075_stag-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
