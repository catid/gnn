#!/usr/bin/env bash
CONFIG=configs/v63_t2b_visit_taskgrad_half_dsg_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t2b-visit_taskgrad_half_dsg-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
