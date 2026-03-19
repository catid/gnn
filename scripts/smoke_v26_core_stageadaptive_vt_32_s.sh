#!/usr/bin/env bash
CONFIG=configs/v26_core_stageadaptive_vt_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-core-stageadaptive_vt-32-s exec "$(dirname "$0")/_smoke_wrapper_single_gpu.sh"
