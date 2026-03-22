#!/usr/bin/env bash
CONFIG=configs/v61_t2b_gate_meta_a_32_s.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-t2b-gate_meta_a-32-s exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
