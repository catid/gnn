#!/usr/bin/env bash
CONFIG=configs/v23_t1_visit_taskgrad_threequarter_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v23-t1-visit_taskgrad_threequarter-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
