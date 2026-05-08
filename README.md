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

For a stronger CPU-only implementation check:

```bash
python scripts/check_imb_cpu.py
```

This additionally verifies IMB special cases, a hand-computed budget formula,
ViT forward/backward, trainable budget gradients, and the ResNet block path.

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

## Experiment Roadmap

Run experiments in this order. The goal is to catch implementation or
configuration issues early, before spending A6000 time on long runs.

### 0. CPU or Login-Node Checks

Run these before using the GPU queue:

```bash
python scripts/smoke_baseline.py
python scripts/check_imb_cpu.py
```

Expected result:

```text
baseline smoke checks passed
all CPU IMB checks passed
```

Do not start full training until both pass.

### 1. GPU Sanity Run

First confirm that CUDA DDP, data loading, logging, and checkpoint directories
work:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_imb.yaml \
  --epochs 1 \
  --max_train_steps 20 \
  --log_interval 5 \
  --save_every_steps 100000
```

Check that:

- training starts on CUDA;
- loss is finite;
- no NCCL/DDP error appears;
- CSV or W&B logging works;
- IMB diagnostics such as `pattern/imb_lambda_mean` and
  `pattern/imb_clip_rate` appear.

### 2. Short CIFAR-10 Pilot

Run short versions of the minimum comparison set:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_linear.yaml \
  --residual_connection linear \
  --epochs 20

torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_orthogonal.yaml \
  --residual_connection orthogonal \
  --orthogonal_method feature \
  --epochs 20

torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_imb.yaml \
  --epochs 20

torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_imb_learned.yaml \
  --epochs 20
```

This stage is not for final accuracy. It checks whether IMB behaves sensibly:

- `imb_lambda_mean` should be between 0 and 1;
- `imb_clip_rate` should not be always 0 or always 1 for all layers unless the
  budget is intentionally extreme;
- validation accuracy should not collapse compared with linear/ORU;
- trainable `tau/kappa` should remain finite and non-negative.

### 3. Budget Sweep on CIFAR-10

Before long runs, sweep a small grid to find reasonable fixed budgets:

```text
tau   in {0.0, 0.01, 0.02, 0.05}
kappa in {0.0, 0.25, 0.5, 1.0}
```

Recommended cheap setting:

```bash
--epochs 50
```

Interpretation:

- `tau=0, kappa=0` should match ORU.
- very large budgets should approach linear residuals.
- useful budgets should show non-trivial clipping and stable accuracy.

Pick 1-2 fixed budget settings for longer CIFAR-100/TinyImageNet runs.

### 4. Main CIFAR-100 Runs

Run the core comparison first on CIFAR-100:

```text
linear
orthogonal
imb fixed
imb learned
```

Recommended:

```text
ViT-S
300 epochs
3 seeds
```

Use seeds such as:

```bash
--seed 0
--seed 1
--seed 2
```

Example:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar100_imb.yaml \
  --seed 0
```

Track:

- best validation accuracy;
- final validation accuracy;
- training loss stability;
- `imb_lambda_mean` by layer and module;
- `imb_clip_rate` by layer and module;
- `imb_radial_excess_mean`;
- `imb_innovation_ratio_mean`.

### 5. TinyImageNet Confirmation

After CIFAR-100 shows non-broken behavior, repeat the core comparison on
TinyImageNet:

```text
linear
orthogonal
imb fixed
imb learned
```

Start with 1 seed. If the trend matches CIFAR-100, expand to 3 seeds.

### 6. Required Strong Baselines

After IMB is working, add baselines that directly address novelty concerns:

- scalar gated ORU:
  `x <- x + u_perp + (1 - g_l) u_parallel`
- ORU -> Linear schedule:
  `--connect_schedule 0:orthogonal,150:linear`
- unified parallel/orthogonal residual family from the ORU appendix.

These are important because IMB must beat or explain more than standard
linear/ORU alone.

### 7. Analysis Figures

For every serious run, produce these plots:

- validation accuracy vs epoch;
- training loss vs epoch;
- `imb_lambda_mean` across depth;
- `imb_clip_rate` across depth;
- `imb_radial_excess_mean` across depth;
- `imb_innovation_ratio_mean` across depth;
- attention and MLP shown separately;
- effective rank or spectral entropy if feature extraction is available.

The most important story to verify is:

```text
early layers: more clipping / more innovation
late layers: more bounded radial memory reinforcement
```

### 8. Stop/Go Criteria

Continue to larger experiments only if:

- IMB does not collapse on CIFAR-10 short runs;
- IMB diagnostics are non-trivial;
- CIFAR-100 accuracy is competitive with ORU and linear;
- the learned/fixed budgets show interpretable layer or module structure.

Pause and debug if:

- `lambda` is always exactly 0 or 1 unintentionally;
- `clip_rate` is saturated for every layer;
- `tau/kappa` explode;
- loss becomes NaN;
- IMB is consistently worse than both linear and ORU across several budget
  settings.

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
