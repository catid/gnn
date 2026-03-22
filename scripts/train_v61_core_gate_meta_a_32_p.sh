#!/usr/bin/env bash
CONFIG=configs/v61_core_gate_meta_a_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-core-gate_meta_a-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
