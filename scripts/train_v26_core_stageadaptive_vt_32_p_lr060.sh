#!/usr/bin/env bash
CONFIG=configs/v26_core_stageadaptive_vt_32_p_lr060.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-core-stageadaptive_vt-32-p-lr060 exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
