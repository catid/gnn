#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v9_clone_selective_long.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v9-clone-selective-long-smoke exec "$(dirname "$0")/_smoke_wrapper.sh"
