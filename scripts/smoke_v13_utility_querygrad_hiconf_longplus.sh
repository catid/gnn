#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v13_utility_querygrad_hiconf_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v13-utility-querygrad-hiconf-smoke TRAIN_STEPS_OVERRIDE=220 exec "$(dirname "$0")/_train_wrapper.sh"
