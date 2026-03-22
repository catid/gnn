#!/usr/bin/env bash
CONFIG=configs/v61_hmix_gate_writers_le2_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-hmix-gate_writers_le2-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
