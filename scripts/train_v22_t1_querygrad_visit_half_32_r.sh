#!/usr/bin/env bash
CONFIG=configs/v22_t1_querygrad_visit_half_32_r.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v22-t1-querygrad_visit_half-32-r exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
