#!/usr/bin/env bash
CONFIG=configs/v23_core_visit_taskgrad_quarter_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v23-core-visit_taskgrad_quarter-32-l exec "$(dirname "$0")/_smoke_wrapper_single_gpu.sh"
