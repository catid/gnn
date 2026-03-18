#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v11_utility_querygrad_condmut_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v11-utility-querygrad-condmut-longplus exec "$(dirname "$0")/_train_wrapper.sh"
