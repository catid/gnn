#!/usr/bin/env bash
CONFIG=configs/v63_t2a_visit_taskgrad_half_ds_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-t2a-visit_taskgrad_half_ds-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
