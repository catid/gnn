#!/usr/bin/env bash
CONFIG=configs/v64_t1r_visit_taskgrad_half_ds_core_runner_up_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t1r-visit_taskgrad_half_ds_core_runner_up-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
