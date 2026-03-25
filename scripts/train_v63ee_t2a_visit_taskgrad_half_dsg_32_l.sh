#!/usr/bin/env bash
CONFIG=configs/v63ee_t2a_visit_taskgrad_half_dsg_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-t2a-visit_taskgrad_half_dsg-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
