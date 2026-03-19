#!/usr/bin/env bash
CONFIG=configs/v19_transfer_t1_full_querygrad_scale.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v19-transfer-t1-full-querygrad-scale exec "$(dirname "$0")/_train_wrapper.sh"
