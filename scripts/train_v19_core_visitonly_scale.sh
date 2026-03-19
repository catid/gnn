#!/usr/bin/env bash
CONFIG=configs/v19_core_visitonly_scale.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v19-core-visitonly-scale exec "$(dirname "$0")/_train_wrapper.sh"
