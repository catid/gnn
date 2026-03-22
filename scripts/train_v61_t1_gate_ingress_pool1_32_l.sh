#!/usr/bin/env bash
CONFIG=configs/v61_t1_gate_ingress_pool1_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-t1-gate_ingress_pool1-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
