#!/usr/bin/env bash
CONFIG=configs/v26_core_visit_taskgrad_0375_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-core-visit_taskgrad_0375-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
