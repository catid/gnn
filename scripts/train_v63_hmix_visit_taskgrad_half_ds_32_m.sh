#!/usr/bin/env bash
CONFIG=configs/v63_hmix_visit_taskgrad_half_ds_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-hmix-visit_taskgrad_half_ds-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
