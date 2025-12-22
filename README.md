# Orthogonal Residual + Radial SDE (Chi)

Geometry-aware residual connections for stable + expressive representations.

This repo started as an Orthogonal Residual Update (ORU) baseline and is reorganized to run **Neural SDE-style residual dynamics** as the *main* experiment:

**Angular (surface)**: orthogonal/tangent update `f_perp(x)`  
**Radial (thickness)**: learnable `(alpha, beta)` controlling radial drift/noise toward a chi / isotropic Gaussian radius.

> Note: we implement the **Euler–Maruyama discretization** directly in PyTorch; `torchsde` is not required for the fixed-depth (ViT/ResNet) setting.

## TL;DR

We replace the usual residual connection

`x <- x + f(x)`

with a geometry-aware update

`x <- x + f_perp(x) + ( radial drift + radial noise ) * u`

where `u = x / (|x| + eps)` and `f_perp(x)` removes the component of `f(x)` parallel to `x`.

### Final update rule (per token / per feature-vector)

Let `r = |x|`, `u = x / (r + eps)`, and `d` be the feature dimension.

```
f_perp(x) = f(x) - <x,f(x)>/(|x|^2+eps) * x
Δr_drift  = α * ((d-1)/(r+eps) - r/σ^2)
Δr_noise  = β * ξ,   ξ ~ N(0,1)
Δx_rad    = (Δr_drift + Δr_noise) * u
```

Final:

`x <- x + f_perp(x) + Δx_rad`

### Why learnable `(alpha,beta)`?

We treat radial shaping as a **soft, diagnostic constraint**:
- if the task wants thin-sphere behavior, it can push `alpha,beta -> 0`
- if the task benefits from controlled norm freedom (gating / confidence / saliency signals), it can keep non-zero radial dynamics

## Repository structure

```
.
├─ connect/                 # residual connect API + logging
├─ models/                  # ViT / ResNet blocks
├─ data/                    # datasets + transforms
├─ configs/                 # YAML configs (toy target: ViT-S/CIFAR10)
│  └─ optional/             # optional schedules/ablations
├─ scripts/                 # train helpers
├─ docs/                    # baseline ORU README kept here
└─ train_classifier.py      # main entrypoint (supervised toy)
```

## Quickstart (ViT-S / CIFAR10)

Install:
```bash
pip install -r requirements.txt
```

### Baselines

- Linear residual:
```bash
bash scripts/train_vit_s_cifar10_linear.sh
```

- Orthogonal residual (ORU):
```bash
bash scripts/train_vit_s_cifar10_orthogonal.sh
```

### Main: SDE-style residual dynamics

```bash
bash scripts/train_vit_s_cifar10_sde.sh
```

Config: `configs/vit_s_cifar10_sde.yaml`
- `residual_connection: sde`
- `sde_trainable: true` (learn `(alpha,beta)` per block/branch)
- `sde_noise_mode: train|always|off`

### Optional (not main): ORU → Linear schedule

This is kept for reference only:
```bash
bash scripts/train_vit_s_cifar10_ortho_to_linear.sh
```

Config: `configs/optional/vit_s_cifar10_ortho_to_linear.yaml`

## What gets logged (for dynamics analysis)

During training, we log per-block activation/connection stats to W&B under `activation/…`:
- `x_norm2`, `f_par_norm2`, `f_ortho_norm2`, `rho_par`, `rho_ortho`, `cos_x_out`
- SDE (when enabled): `sde_alpha`, `sde_beta`, `sde_r_mean`, `sde_r_std`, `sde_entropy_mean`, `sde_restoring_mean`, `sde_rho_rad`, …

These are meant to answer questions like:
- does the model keep a thin sphere (orthogonal-only) or develop thickness (SDE)?
- which layers/branches keep `alpha,beta` active vs collapse them to ~0?
- how does radial injection trade off with downstream accuracy/stability?

You can also inspect learned `alpha/beta` scalars from a checkpoint:
```bash
python scripts/inspect_sde_params.py --checkpoint path/to/checkpoints/0005000.pt
```

## Notes

- If you need the original ORU baseline README, see `docs/ORTHO_RESIDUAL_BASELINE.md`.
- This repo’s “SDE” claim is **discretized dynamics** (Euler–Maruyama residual blocks), not a continuous-time solver setup.
