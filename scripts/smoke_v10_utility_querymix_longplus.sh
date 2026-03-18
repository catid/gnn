#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v10_utility_querymix_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v10-utility-querymix-longplus exec "$(dirname "$0")/_smoke_wrapper.sh"
