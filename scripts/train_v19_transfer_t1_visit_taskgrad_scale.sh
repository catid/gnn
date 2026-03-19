#!/usr/bin/env bash
CONFIG=configs/v19_transfer_t1_visit_taskgrad_scale.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v19-transfer-t1-visit-taskgrad-scale exec "$(dirname "$0")/_train_wrapper.sh"
