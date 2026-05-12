#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
EPOCHS_SHORT="${EPOCHS_SHORT:-50}"

run() {
  local name="$1"
  shift
  echo "=== ${name} ==="
  CUDA_VISIBLE_DEVICES="${GPU_ID}" torchrun --nproc-per-node 1 train_classifier.py "$@"
}

# 1) Prove the implemented IMB fallback matches the ORU training setting.
run "cifar10_imb_oru_fallback" \
  --config_file configs/vit_s_cifar10_imb.yaml \
  --imb_tau 0.0 \
  --imb_kappa 0.0 \
  --seed "${SEED}" \
  --epochs "${EPOCHS_SHORT}"

# 2) Cheap learned-budget variant: ORU-near init, slow budget learning, budget L1.
run "cifar10_imb_learned_oru_near" \
  --config_file configs/vit_s_cifar10_imb_learned_oru_near.yaml \
  --seed "${SEED}" \
  --epochs "${EPOCHS_SHORT}"

# 3) Cheap learned depth schedule: early ORU-like initialization, late memory-permissive initialization,
#    but tau/kappa remain trainable.
run "cifar10_imb_learned_depth_scheduled" \
  --config_file configs/vit_s_cifar10_imb_learned_scheduled.yaml \
  --seed "${SEED}" \
  --epochs "${EPOCHS_SHORT}"
