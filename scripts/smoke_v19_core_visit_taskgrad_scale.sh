#!/usr/bin/env bash
CONFIG=configs/v19_core_visit_taskgrad_scale.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v19-core-visit-taskgrad-scale exec "$(dirname "$0")/_smoke_wrapper.sh"
