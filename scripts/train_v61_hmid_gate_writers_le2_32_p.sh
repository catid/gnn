#!/usr/bin/env bash
CONFIG=configs/v61_hmid_gate_writers_le2_32_p.yaml DEFAULT_SEED=1234 DEFAULT_RUN_NAME=v61-hmid-gate_writers_le2-32-p exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"
