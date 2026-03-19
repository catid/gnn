#!/usr/bin/env bash
CONFIG=configs/v26_core_visit_taskgrad_0625_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v26-core-visit_taskgrad_0625-32-s exec "$(dirname "$0")/_smoke_wrapper_single_gpu.sh"
