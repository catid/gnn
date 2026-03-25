#!/usr/bin/env bash
CONFIG=configs/v64_t2a_visit_taskgrad_half_ds_core_runner_up_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t2a-visit_taskgrad_half_ds_core_runner_up-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
