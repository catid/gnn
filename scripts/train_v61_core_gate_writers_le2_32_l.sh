#!/usr/bin/env bash
CONFIG=configs/v61_core_gate_writers_le2_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-core-gate_writers_le2-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
