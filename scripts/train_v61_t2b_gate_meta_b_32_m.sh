#!/usr/bin/env bash
CONFIG=configs/v61_t2b_gate_meta_b_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-t2b-gate_meta_b-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
