#!/usr/bin/env bash
CONFIG=configs/v26_core_visitonly_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-core-visitonly-32-s exec "$(dirname "$0")/_smoke_wrapper_single_gpu.sh"
