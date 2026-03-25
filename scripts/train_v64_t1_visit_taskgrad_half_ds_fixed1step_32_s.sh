#!/usr/bin/env bash
CONFIG=configs/v64_t1_visit_taskgrad_half_ds_fixed1step_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-t1-visit_taskgrad_half_ds_fixed1step-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
