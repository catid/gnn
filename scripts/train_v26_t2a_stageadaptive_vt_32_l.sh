#!/usr/bin/env bash
CONFIG=configs/v26_t2a_stageadaptive_vt_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-t2a-stageadaptive_vt-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
