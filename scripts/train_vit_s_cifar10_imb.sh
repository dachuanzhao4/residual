#!/usr/bin/env bash
set -euo pipefail

# NGPU="${NGPU:-1}"
# CONFIG="${1:-configs/vit_s_cifar10_imb.yaml}"

# torchrun --nproc-per-node "${NGPU}" train_classifier.py --config_file "${CONFIG}"


CUDA_VISIBLE_DEVICES=2 torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_imb.yaml


CUDA_VISIBLE_DEVICES=2 torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_imb_learned.yaml


CUDA_VISIBLE_DEVICES=2 torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar100_imb.yaml

