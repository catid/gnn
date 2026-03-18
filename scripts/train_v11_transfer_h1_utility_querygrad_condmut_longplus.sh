#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v11_transfer_h1_utility_querygrad_condmut_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v11-transfer-h1-utility-querygrad-condmut-longplus exec "$(dirname "$0")/_train_wrapper.sh"
