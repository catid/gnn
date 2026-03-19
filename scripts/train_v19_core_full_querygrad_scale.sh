#!/usr/bin/env bash
CONFIG=configs/v19_core_full_querygrad_scale.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v19-core-full-querygrad-scale exec "$(dirname "$0")/_train_wrapper.sh"
