#!/usr/bin/env bash
CONFIG=configs/v64_hmix_visit_taskgrad_half_ds_auxanneal_025_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-hmix-visit_taskgrad_half_ds_auxanneal_025-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
