#!/usr/bin/env bash
CONFIG=configs/v64_t2b_visit_taskgrad_half_ds_auxanneal_050_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2b-visit_taskgrad_half_ds_auxanneal_050-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
