#!/usr/bin/env bash
CONFIG=configs/v24_t2a_visit_taskgrad_threeeighth_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v24-t2a-visit_taskgrad_threeeighth-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
