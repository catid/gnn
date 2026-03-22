#!/usr/bin/env bash
CONFIG=configs/v61_core_gate_ingress_pool1_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-core-gate_ingress_pool1-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
