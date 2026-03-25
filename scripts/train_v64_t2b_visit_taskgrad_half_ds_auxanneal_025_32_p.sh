#!/usr/bin/env bash
CONFIG=configs/v64_t2b_visit_taskgrad_half_ds_auxanneal_025_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2b-visit_taskgrad_half_ds_auxanneal_025-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
