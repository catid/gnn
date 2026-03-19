#!/usr/bin/env bash
CONFIG=configs/v23_t2a_visit_taskgrad_quarter_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v23-t2a-visit_taskgrad_quarter-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
