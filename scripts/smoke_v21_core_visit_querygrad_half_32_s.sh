#!/usr/bin/env bash
CONFIG=configs/v21_core_visit_querygrad_half_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v21-core-visit_querygrad_half-32-s exec "$(dirname "$0")/_smoke_wrapper_single_gpu.sh"
