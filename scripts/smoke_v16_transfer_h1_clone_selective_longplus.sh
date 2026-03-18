#!/usr/bin/env bash
CONFIG=configs/v16_transfer_h1_clone_selective_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v16-transfer-h1-clone-selective-smoke TRAIN_STEPS_OVERRIDE=220 exec "$(dirname "$0")/_train_wrapper.sh"
