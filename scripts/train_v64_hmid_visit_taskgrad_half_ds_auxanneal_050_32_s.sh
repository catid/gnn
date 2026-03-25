#!/usr/bin/env bash
CONFIG=configs/v64_hmid_visit_taskgrad_half_ds_auxanneal_050_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v64-hmid-visit_taskgrad_half_ds_auxanneal_050-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
