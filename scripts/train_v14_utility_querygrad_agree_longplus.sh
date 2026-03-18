#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v14_utility_querygrad_agree_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v14-utility-querygrad-agree-longplus exec "$(dirname "$0")/_train_wrapper.sh"
