#!/usr/bin/env bash
CONFIG=configs/v63ee_hmix_visit_taskgrad_half_dsg_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63ee-hmix-visit_taskgrad_half_dsg-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
