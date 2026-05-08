import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connect import connect
from models.ortho_models import OrthoBlock
from models.vit import Classifier


def check_connect(method: str, orthogonal_method: str, eps: float) -> None:
    torch.manual_seed(0)
    x = torch.randn(4, 17, 64)
    u = torch.randn_like(x)
    kwargs = {}
    if method == "imb":
        kwargs = {"imb_tau": torch.tensor([0.0]), "imb_kappa": torch.tensor([0.0])}
    y, stats = connect(
        x,
        u,
        method=method,
        orthogonal_method=orthogonal_method,
        dim=-1,
        eps=eps,
        **kwargs,
    )
    if method == "linear":
        torch.testing.assert_close(y, x + u)
        print("connect(linear): ok")
        return

    if method == "orthogonal":
        delta = y - x
        dot = (delta * x).sum(dim=-1).abs().max().item()
        if dot > 1e-4:
            raise AssertionError(f"orthogonal residual is not orthogonal enough: max |dot|={dot}")
        print(f"connect(orthogonal/{orthogonal_method}): ok, max |dot(delta,x)|={dot:.3e}")
    if stats.f_par is None or stats.f_ortho is None:
        raise AssertionError("orthogonal stats did not include parallel/orthogonal components")
    if method == "imb":
        if "imb_lambda_mean" not in stats.extras:
            raise AssertionError("IMB stats did not include lambda diagnostics")
        delta = y - x
        par_norm = stats.f_par.norm(dim=-1)
        raw_par_norm = stats.extras["imb_par_norm_raw_mean"].item()
        if par_norm.mean().item() > raw_par_norm + 1e-4:
            raise AssertionError("IMB increased the parallel component norm")
        print(f"connect(imb/{orthogonal_method}): ok, lambda_mean={stats.extras['imb_lambda_mean'].item():.3f}")


def build_tiny_vit(method: str, orthogonal_method: str, num_classes: int) -> Classifier:
    imb_kwargs = {}
    if method == "imb":
        imb_kwargs = {"imb_tau": 0.05, "imb_kappa": 0.5}
    return Classifier(
        img_size=32,
        dim=64,
        patch_size=4,
        num_heads=4,
        num_layers=2,
        in_chans=3,
        num_classes=num_classes,
        class_token=True,
        reg_tokens=0,
        pos_embed="learn",
        block_class=OrthoBlock,
        residual_connection=method,
        orthogonal_method=orthogonal_method,
        residual_eps=1e-6,
        residual_pattern="default",
        residual_rescale_mode="scalar",
        drop_path=0.0,
        mlp_dropout=0.0,
        log_interval=1,
        log_activations=True,
        gradient_checkpointing="none",
        **imb_kwargs,
    )


def check_model_forward(method: str, orthogonal_method: str, batch_size: int) -> None:
    torch.manual_seed(1)
    model = build_tiny_vit(method, orthogonal_method, num_classes=10)
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32)
    with torch.no_grad():
        logits = model(x)
    if logits.shape != (batch_size, 10):
        raise AssertionError(f"unexpected logits shape: {tuple(logits.shape)}")
    stats = model.pop_stats()
    print(f"tiny ViT forward ({method}): ok, logits={tuple(logits.shape)}, stats={len(stats)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orthogonal_method", default="feature", choices=["feature", "global"])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eps", type=float, default=1e-6)
    args = parser.parse_args()

    check_connect("linear", args.orthogonal_method, args.eps)
    check_connect("orthogonal", args.orthogonal_method, args.eps)
    check_connect("imb", args.orthogonal_method, args.eps)
    check_model_forward("linear", args.orthogonal_method, args.batch_size)
    check_model_forward("orthogonal", args.orthogonal_method, args.batch_size)
    check_model_forward("imb", args.orthogonal_method, args.batch_size)
    print("baseline smoke checks passed")


if __name__ == "__main__":
    main()
