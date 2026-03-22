#!/usr/bin/env bash
CONFIG=configs/v53_t2c_visit_taskgrad_half_32_xl.yaml DEFAULT_SEED=172234 DEFAULT_RUN_NAME=v53-t2c-visit_taskgrad_half-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
