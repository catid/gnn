#!/usr/bin/env bash
CONFIG=configs/v24_t1_visit_taskgrad_fiveeighth_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v24-t1-visit_taskgrad_fiveeighth-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
