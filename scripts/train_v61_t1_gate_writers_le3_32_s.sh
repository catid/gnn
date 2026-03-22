#!/usr/bin/env bash
CONFIG=configs/v61_t1_gate_writers_le3_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-t1-gate_writers_le3-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
