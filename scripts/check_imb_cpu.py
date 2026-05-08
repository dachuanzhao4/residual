import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from connect import connect
from models.ortho_models import OrthoBlock
from models.preactresnet import PreActBasic
from models.vit import Classifier


def assert_close(a, b, msg, **kwargs):
    try:
        torch.testing.assert_close(a, b, **kwargs)
    except AssertionError as exc:
        raise AssertionError(msg) from exc


def check_imb_special_cases():
    torch.manual_seed(10)
    x = torch.randn(3, 5, 16)
    u = torch.randn_like(x)

    y_ortho, _ = connect(x, u, method="orthogonal", orthogonal_method="feature")
    y_imb_zero, stats_zero = connect(
        x,
        u,
        method="imb",
        orthogonal_method="feature",
        imb_tau=torch.tensor([0.0]),
        imb_kappa=torch.tensor([0.0]),
    )
    assert_close(y_imb_zero, y_ortho, "IMB with zero budget should match ORU", atol=1e-6, rtol=1e-6)
    assert stats_zero.extras["imb_lambda_mean"].item() < 1e-6

    y_linear, _ = connect(x, u, method="linear", orthogonal_method="feature")
    y_imb_large, stats_large = connect(
        x,
        u,
        method="imb",
        orthogonal_method="feature",
        imb_tau=torch.tensor([1e6]),
        imb_kappa=torch.tensor([1e6]),
    )
    assert_close(y_imb_large, y_linear, "IMB with huge budget should match linear residual", atol=1e-5, rtol=1e-6)
    assert stats_large.extras["imb_lambda_mean"].item() > 1.0 - 1e-6
    print("IMB special cases: ok")


def check_imb_budget_formula():
    x = torch.tensor([[[2.0, 0.0]]])
    u = torch.tensor([[[3.0, 4.0]]])
    tau = torch.tensor([0.25])
    kappa = torch.tensor([0.5])
    y, stats = connect(
        x,
        u,
        method="imb",
        orthogonal_method="feature",
        imb_tau=tau,
        imb_kappa=kappa,
    )

    u_parallel = torch.tensor([[[3.0, 0.0]]])
    u_perp = torch.tensor([[[0.0, 4.0]]])
    radial_budget = tau * torch.tensor([[[2.0]]]) + kappa * torch.tensor([[[4.0]]])
    lam = radial_budget / torch.tensor([[[3.0]]])
    expected = x + u_perp + lam * u_parallel
    assert_close(y, expected, "IMB closed-form budget formula mismatch", atol=1e-6, rtol=1e-6)
    assert abs(stats.extras["imb_lambda_mean"].item() - lam.item()) < 1e-6
    print("IMB budget formula: ok")


def check_vit_forward_backward(trainable: bool):
    torch.manual_seed(11)
    kwargs = {
        "imb_trainable": trainable,
        "imb_tau_init": 0.02,
        "imb_kappa_init": 0.5,
    } if trainable else {
        "imb_tau": 0.02,
        "imb_kappa": 0.5,
    }
    model = Classifier(
        img_size=32,
        dim=64,
        patch_size=4,
        num_heads=4,
        num_layers=2,
        in_chans=3,
        num_classes=10,
        class_token=True,
        block_class=OrthoBlock,
        residual_connection="imb",
        orthogonal_method="feature",
        log_interval=1,
        log_activations=True,
        **kwargs,
    )
    model.train()
    x = torch.randn(4, 3, 32, 32)
    y = torch.tensor([0, 1, 2, 3])
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    stats = model.pop_stats()
    if len(stats) != 4:
        raise AssertionError(f"expected 4 activation stats, got {len(stats)}")
    if "imb_lambda_mean" not in stats[0].extras:
        raise AssertionError("missing IMB diagnostics in ViT stats")
    if trainable:
        grads = [
            p.grad
            for name, p in model.named_parameters()
            if "imb_raw" in name
        ]
        if not grads or any(g is None for g in grads):
            raise AssertionError("trainable IMB parameters did not receive gradients")
    print(f"ViT IMB forward/backward trainable={trainable}: ok")


def check_resnet_block_forward():
    torch.manual_seed(12)
    block = PreActBasic(
        8,
        8,
        stride=1,
        residual_connection="imb",
        orthogonal_method="feature",
        imb_tau=0.02,
        imb_kappa=0.5,
        log_interval=1,
        log_activations=True,
    )
    block.train()
    x = torch.randn(2, 8, 16, 16)
    y = block(x)
    if y.shape != x.shape:
        raise AssertionError(f"unexpected ResNet block output shape: {tuple(y.shape)}")
    loss = y.square().mean()
    loss.backward()
    stats = block.pop_stats()
    if len(stats) != 1 or "imb_lambda_mean" not in stats[0].extras:
        raise AssertionError("missing IMB diagnostics in ResNet block stats")
    print("ResNet block IMB forward/backward: ok")


def main():
    check_imb_special_cases()
    check_imb_budget_formula()
    check_vit_forward_backward(trainable=False)
    check_vit_forward_backward(trainable=True)
    check_resnet_block_forward()
    print("all CPU IMB checks passed")


if __name__ == "__main__":
    main()
