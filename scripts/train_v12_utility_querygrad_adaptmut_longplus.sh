#!/usr/bin/env bash
set -euo pipefail
CONFIG=configs/v12_utility_querygrad_adaptmut_longplus.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v12-utility-querygrad-adaptmut-longplus exec "$(dirname "$0")/_train_wrapper.sh"
