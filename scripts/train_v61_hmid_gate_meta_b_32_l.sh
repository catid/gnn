#!/usr/bin/env bash
CONFIG=configs/v61_hmid_gate_meta_b_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-hmid-gate_meta_b-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
