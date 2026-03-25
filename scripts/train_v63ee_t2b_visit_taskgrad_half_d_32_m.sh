#!/usr/bin/env bash
CONFIG=configs/v63ee_t2b_visit_taskgrad_half_d_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t2b-visit_taskgrad_half_d-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
