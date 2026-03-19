#!/usr/bin/env bash
CONFIG=configs/v19_transfer_t1_visitonly_scale.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v19-transfer-t1-visitonly-scale exec "$(dirname "$0")/_smoke_wrapper.sh"
