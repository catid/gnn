#!/usr/bin/env bash
CONFIG=configs/v64_hmid_visit_taskgrad_half_ds_fixed1step_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-hmid-visit_taskgrad_half_ds_fixed1step-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
