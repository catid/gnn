#!/usr/bin/env bash
CONFIG=configs/v37_t2a_visit_taskgrad_half_agree_mutate_z00_m075_f0375_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v37-t2a-visit_taskgrad_half_agree_mutate_z00_m075_f0375-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
