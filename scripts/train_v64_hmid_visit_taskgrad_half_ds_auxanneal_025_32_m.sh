#!/usr/bin/env bash
CONFIG=configs/v64_hmid_visit_taskgrad_half_ds_auxanneal_025_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-hmid-visit_taskgrad_half_ds_auxanneal_025-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
