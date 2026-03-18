#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v13_transfer_h1_utility_querygrad_hiconf_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v13-transfer-h1-utility-querygrad-hiconf-longplus exec "$(dirname "$0")/_train_wrapper.sh"
