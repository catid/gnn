#!/usr/bin/env bash
CONFIG=configs/v61_t2a_gate_writers_le2_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-t2a-gate_writers_le2-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
