#!/usr/bin/env bash
CONFIG=configs/v22_core_querygrad_visit_quarter_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v22-core-querygrad_visit_quarter-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
