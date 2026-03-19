#!/usr/bin/env bash
CONFIG=configs/v26_t2c_stageadaptive_vt_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-t2c-stageadaptive_vt-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
