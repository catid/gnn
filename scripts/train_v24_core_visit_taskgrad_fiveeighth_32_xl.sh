#!/usr/bin/env bash
CONFIG=configs/v24_core_visit_taskgrad_fiveeighth_32_xl.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v24-core-visit_taskgrad_fiveeighth-32-xl exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
