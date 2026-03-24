#!/usr/bin/env bash
CONFIG=configs/v63_t1_visit_taskgrad_half_dsg_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t1-visit_taskgrad_half_dsg-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
