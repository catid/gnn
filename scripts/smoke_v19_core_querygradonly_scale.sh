#!/usr/bin/env bash
CONFIG=configs/v19_core_querygradonly_scale.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v19-core-querygradonly-scale exec "$(dirname "$0")/_smoke_wrapper.sh"
