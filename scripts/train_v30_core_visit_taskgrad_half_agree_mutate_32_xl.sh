#!/usr/bin/env bash
CONFIG=configs/v30_core_visit_taskgrad_half_agree_mutate_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v30-core-visit_taskgrad_half_agree_mutate-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
