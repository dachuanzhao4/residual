# Innovation-Memory Budgeted Residuals

This repository extends the official Orthogonal Residual Update (ORU) codebase
with **Innovation-Memory Budgeted Residuals (IMB-Res)**, a constrained geometric
residual update for Transformer and ResNet-style models.

The project starts from the observation that the component of a residual update
parallel to the current residual stream is not always redundant. It can also
carry useful memory retention, semantic reinforcement, or confidence evidence.
IMB-Res therefore clips only **excessive radial updates**, while preserving all
orthogonal innovation.

Upstream ORU implementation:

https://github.com/BootsofLagrangian/ortho-residual

## Method

For a residual stream `x` and module output `u = F_l(x)`, decompose the update
into parallel and orthogonal components:

```text
u_parallel = <u, x> / (||x||^2 + eps) * x
u_perp     = u - u_parallel
```

Standard residuals add both components:

```text
x <- x + u_perp + u_parallel
```

ORU removes the parallel component:

```text
x <- x + u_perp
```

IMB-Res keeps the full orthogonal component, then keeps as much parallel evidence
as allowed by an innovation-memory budget:

```text
R_allow = tau * ||x|| + kappa * ||u_perp||
lambda  = min(1, R_allow / (||u_parallel|| + eps))
x       <- x + u_perp + lambda * u_parallel
```

Here:

- `tau` is the memory budget coefficient.
- `kappa` controls how much radial reinforcement is supported by orthogonal
  innovation.
- `lambda` is token/state dependent, not a free layer-wise interpolation gate.

Special cases:

- `tau, kappa -> infinity`: standard residual.
- `tau = 0, kappa = 0`: ORU.
- intermediate budgets: bounded radial memory reinforcement.

## What Was Added

The new residual method is available as:

```text
residual_connection: imb
```

Main implementation points:

- `connect/__init__.py`: IMB update and diagnostics.
- `models/ortho_models.py`: ViT block support for fixed or trainable IMB budgets.
- `models/preactresnet.py`: ResNet block support for fixed or trainable IMB budgets.
- `train_classifier.py`: CLI arguments and optimizer parameter groups.
- `configs/*_imb*.yaml`: ready-to-run IMB experiment configs.
- `scripts/smoke_baseline.py`: CPU-friendly smoke test for linear, ORU, and IMB.

## Installation

```bash
pip install -r requirements.txt
pip install torchvision
```

The official training script expects CUDA DDP/NCCL. A CPU-only machine can run
the smoke test, but full training should be run on a GPU.

## Smoke Test

```bash
python scripts/smoke_baseline.py
```

This checks:

- linear residual equals `x + u`;
- ORU produces an update orthogonal to `x`;
- IMB produces a clipped parallel component and emits diagnostics;
- tiny ViT forward passes work for `linear`, `orthogonal`, and `imb`.

## Training

### Baselines

Linear residual on CIFAR-10:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_linear.yaml \
  --residual_connection linear
```

Orthogonal Residual Update on CIFAR-10:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_orthogonal.yaml \
  --residual_connection orthogonal \
  --orthogonal_method feature
```

### IMB-Res

Fixed-budget IMB on CIFAR-10:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_imb.yaml
```

Learnable-budget IMB on CIFAR-10:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_imb_learned.yaml
```

Fixed-budget IMB on CIFAR-100:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar100_imb.yaml
```

The helper script is also available:

```bash
bash scripts/train_vit_s_cifar10_imb.sh
```

## Useful CLI Arguments

```text
--residual_connection imb
--imb_tau 0.02
--imb_kappa 0.5
--imb_trainable
--imb_tau_init 0.02
--imb_kappa_init 0.5
--imb_param_lr_mult 0.1
--imb_param_weight_decay 0.0
```

For fixed budgets, set `--imb_tau` and `--imb_kappa`.

For learnable budgets, use `--imb_trainable` with initial values
`--imb_tau_init` and `--imb_kappa_init`.

## Diagnostics

IMB-Res logs the existing ORU activation statistics plus IMB-specific values:

- `x_norm2`
- `f_par_norm2`
- `f_ortho_norm2`
- `rho_par`
- `rho_ortho`
- `cos_x_out`
- `pattern/imb_tau`
- `pattern/imb_kappa`
- `pattern/imb_lambda_mean`
- `pattern/imb_clip_rate`
- `pattern/imb_radial_budget_mean`
- `pattern/imb_radial_excess_mean`
- `pattern/imb_innovation_ratio_mean`
- `pattern/imb_par_norm_raw_mean`
- `pattern/imb_ortho_norm_raw_mean`

Important plots:

- `imb_lambda_mean` across depth;
- `imb_clip_rate` across depth;
- `imb_radial_excess_mean` for identifying radial redundancy;
- `imb_innovation_ratio_mean` for tracking update geometry;
- attention vs MLP dynamics separately.

## Suggested Experiment Matrix

Start small:

```text
CIFAR-10, ViT-S, 20-50 epochs
linear / orthogonal / imb fixed / imb learned
```

Then run the main project matrix:

```text
CIFAR-100 and TinyImageNet
ViT-S
linear / orthogonal / scalar gated ORU / IMB fixed / IMB learned
3 seeds
```

For stronger research evidence, add:

- ORU -> Linear schedule;
- the unified residual family baseline from the ORU paper appendix;
- a small causal Transformer on TinyStories or WikiText-103.

## Resource Notes

A single 48GB A6000 is enough for the main ViT-S CIFAR-100 and TinyImageNet
experiments. Free Colab is useful for smoke tests and short CIFAR-10 sanity
runs, but it is not reliable for full 300-epoch multi-seed experiments.

## Attribution

This repository is built on the official implementation of:

```bibtex
@article{oh2025revisiting,
  title={Revisiting Residual Connections: Orthogonal Updates for Stable and Efficient Deep Networks},
  author={Giyeong Oh and Woohyun Cho and Siyeol Kim and Suhwan Choi and Youngjae Yu},
  year={2025},
  journal={arXiv preprint arXiv:2505.11881},
  eprint={2505.11881},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2505.11881}
}
```

