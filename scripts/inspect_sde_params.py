#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect learned SDE (alpha,beta) parameters from a checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a .pt checkpoint saved by train_classifier.py")
    parser.add_argument("--device", type=str, default="cpu", help="Device for loading tensors (cpu/cuda)")
    return parser.parse_args()


def softplus(x: torch.Tensor) -> torch.Tensor:
    return F.softplus(x)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    state = ckpt.get("model", ckpt)

    raw_alpha_keys = [k for k in state.keys() if k.endswith("sde_raw_alpha")]
    if not raw_alpha_keys:
        print("No trainable SDE parameters found (no '*sde_raw_alpha' keys).")
        return

    rows = []
    for k in sorted(raw_alpha_keys):
        raw_alpha = state[k].detach().float().reshape(-1)[0]
        alpha = softplus(raw_alpha)
        beta_key = k.replace("sde_raw_alpha", "sde_raw_beta_scale")
        if beta_key in state:
            beta_scale = state[beta_key].detach().float().reshape(-1)[0]
            beta = torch.sqrt(2.0 * alpha) * torch.exp(beta_scale)
        else:
            beta_scale = torch.tensor(float("nan"))
            beta = torch.sqrt(2.0 * alpha)

        rows.append((k, alpha.item(), beta.item(), beta_scale.item()))

    max_key = max(len(k) for k, *_ in rows)
    header = f"{'param':<{max_key}}  alpha        beta         beta_scale"
    print(header)
    print("-" * len(header))
    for k, alpha, beta, beta_scale in rows:
        beta_scale_str = f"{beta_scale: .6e}" if math.isfinite(beta_scale) else "    (coupled)"
        print(f"{k:<{max_key}}  {alpha: .6e}  {beta: .6e}  {beta_scale_str}")


if __name__ == "__main__":
    main()

