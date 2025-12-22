#!/usr/bin/env bash
set -euo pipefail

NGPU="${NGPU:-1}"
CONFIG="${1:-configs/optional/vit_s_cifar10_ortho_to_linear.yaml}"

torchrun --nproc-per-node "${NGPU}" train_classifier.py --config_file "${CONFIG}"
