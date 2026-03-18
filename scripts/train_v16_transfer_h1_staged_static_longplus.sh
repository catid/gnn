#!/usr/bin/env bash
CONFIG=configs/v16_transfer_h1_staged_static_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v16-transfer-h1-staged-static-longplus exec "$(dirname "$0")/_train_wrapper.sh"
