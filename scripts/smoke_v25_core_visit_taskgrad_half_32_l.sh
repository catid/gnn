#!/usr/bin/env bash
CONFIG=configs/v25_core_visit_taskgrad_half_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v25-core-visit_taskgrad_half-32-l exec "$(dirname "$0")/_smoke_wrapper_single_gpu.sh"
