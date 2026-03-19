#!/usr/bin/env bash
CONFIG=configs/v26_t2c_visit_taskgrad_0375_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-t2c-visit_taskgrad_0375-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
