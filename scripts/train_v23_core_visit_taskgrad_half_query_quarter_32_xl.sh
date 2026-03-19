#!/usr/bin/env bash
CONFIG=configs/v23_core_visit_taskgrad_half_query_quarter_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v23-core-visit_taskgrad_half_query_quarter-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
