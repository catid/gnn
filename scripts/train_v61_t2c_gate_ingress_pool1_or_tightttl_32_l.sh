#!/usr/bin/env bash
CONFIG=configs/v61_t2c_gate_ingress_pool1_or_tightttl_32_l.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-t2c-gate_ingress_pool1_or_tightttl-32-l exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
