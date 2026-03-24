#!/usr/bin/env bash
CONFIG=configs/v63_t1r_visit_taskgrad_half_b_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t1r-visit_taskgrad_half_b-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
