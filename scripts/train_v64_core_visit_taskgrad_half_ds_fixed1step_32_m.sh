#!/usr/bin/env bash
CONFIG=configs/v64_core_visit_taskgrad_half_ds_fixed1step_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-core-visit_taskgrad_half_ds_fixed1step-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
