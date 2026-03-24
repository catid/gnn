#!/usr/bin/env bash
CONFIG=configs/v63_core_visitonly_ds_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v63-core-visitonly_ds-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
