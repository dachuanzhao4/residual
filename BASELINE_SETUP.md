# ORU Baseline Setup

Official repository for the first PDF:

https://github.com/BootsofLagrangian/ortho-residual

The paper baseline is Orthogonal Residual Update (ORU), comparing standard
linear residual updates against orthogonal residual updates:

```text
linear:      x <- x + u
orthogonal:  x <- x + u_perp
```

where `u = F_l(x)` and `u_perp` is obtained by subtracting the projection of
`u` onto the current residual stream `x`.

## Current Local Environment

This workspace currently has CPU-only PyTorch:

```text
torch 2.11.0+cpu
torchvision 0.26.0+cpu
```

The official `train_classifier.py` currently assumes CUDA DDP/NCCL, so full
training should be run on a GPU environment. A CPU smoke test is provided to
verify that the official model and residual connection code paths are usable.

## Smoke Test

```powershell
& 'C:\Users\pc\AppData\Local\Programs\Python\Python313\python.exe' scripts\smoke_baseline.py
```

This checks:

- `connect(linear)` equals `x + u`
- `connect(orthogonal)` produces an update orthogonal to `x`
- tiny ViT forward pass works with both `linear` and `orthogonal` connections

## GPU Baseline Commands

Standard residual ViT-S on CIFAR-10:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_linear.yaml \
  --residual_connection linear
```

Orthogonal residual ViT-S on CIFAR-10:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_orthogonal.yaml \
  --residual_connection orthogonal \
  --orthogonal_method feature
```

Standard residual ViT-S on CIFAR-100:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar100.yaml \
  --residual_connection linear
```

Orthogonal residual ViT-S on CIFAR-100:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar100.yaml \
  --residual_connection orthogonal \
  --orthogonal_method feature
```

For proposal work, treat these as the initial baselines before adding scalar
gated ORU or Innovation-Memory Budgeted Residuals.

## IMB-Res Prototype

This workspace also includes a first implementation of proposal2 under:

```text
residual_connection: imb
```

The implemented update is:

```text
u_parallel = Proj_x(u)
u_perp = u - u_parallel
R_allow = tau * ||x|| + kappa * ||u_perp||
lambda = min(1, R_allow / (||u_parallel|| + eps))
x <- x + u_perp + lambda * u_parallel
```

Fixed-budget CIFAR-10:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_imb.yaml
```

Learnable-budget CIFAR-10:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar10_imb_learned.yaml
```

Fixed-budget CIFAR-100:

```bash
torchrun --nproc-per-node 1 train_classifier.py \
  --config_file configs/vit_s_cifar100_imb.yaml
```

IMB logs additional diagnostics under the existing activation logging path:

- `pattern/imb_tau`
- `pattern/imb_kappa`
- `pattern/imb_lambda_mean`
- `pattern/imb_clip_rate`
- `pattern/imb_radial_budget_mean`
- `pattern/imb_radial_excess_mean`
- `pattern/imb_innovation_ratio_mean`
- `pattern/imb_par_norm_raw_mean`
- `pattern/imb_ortho_norm_raw_mean`
