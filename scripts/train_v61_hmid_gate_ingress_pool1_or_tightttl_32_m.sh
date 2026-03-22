#!/usr/bin/env bash
CONFIG=configs/v61_hmid_gate_ingress_pool1_or_tightttl_32_m.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-hmid-gate_ingress_pool1_or_tightttl-32-m exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
