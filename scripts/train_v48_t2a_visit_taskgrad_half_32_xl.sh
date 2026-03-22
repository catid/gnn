#!/usr/bin/env bash
CONFIG=configs/v48_t2a_visit_taskgrad_half_32_xl.yaml DEFAULT_SEED=130234 DEFAULT_RUN_NAME=v48-t2a-visit_taskgrad_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
