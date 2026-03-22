#!/usr/bin/env bash
CONFIG=configs/v30_t1_visit_taskgrad_half_hiconf_mutate_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v30-t1-visit_taskgrad_half_hiconf_mutate-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
