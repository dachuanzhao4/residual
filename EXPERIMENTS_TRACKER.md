# Innovation-Memory Budgeted Residuals (IMB-Res) - Experiments Tracker

This document tracks the results of the baseline and IMB-Res variant experiments run on ViT-S models.

## Experiment Results

| ID | Configuration | Dataset | Val Acc@1 | Val Acc@5 | Note |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `002` | Baseline: Linear Residual | CIFAR-10 | 89.33% | 99.56% | Standard ViT-S baseline |
| `003` | Baseline: Orthogonal Residual (ORU) | CIFAR-10 | 90.31% | 99.65% | Removes parallel component |
| `004` | Variant 1: IMB-Res Fixed Budget | CIFAR-10 | 89.86% | 99.58% | Clips excessive radial updates |
| `005` | Variant 2: IMB-Res Learnable Budget | CIFAR-10 | 89.20% | 99.61% | Learns $\tau$ and $\kappa$ budgets |
| `006` | Variant 3: IMB-Res Fixed Budget | CIFAR-100 | 71.97% | 92.22% | IMB-Res on more complex dataset |

*(Results are taken from the final evaluation after 300 epochs of training on a single A6000 GPU).*

## Findings and Analysis

1. **Orthogonal Residuals Lead on CIFAR-10**: The Orthogonal Residual Update (ORU) (`003`) achieves the highest top-1 validation accuracy (**90.31%**) among the CIFAR-10 runs, outperforming the standard linear residual baseline (**89.33%**) by a margin of ~1.0%. This confirms the original paper's finding that eliminating the parallel component entirely can improve performance on this specific dataset.

2. **IMB-Res Fixed Budget vs Linear**: The fixed-budget Innovation-Memory Budgeted Residual (`004`) achieves **89.86%** accuracy. While it trails slightly behind the pure orthogonal update, it noticeably outperforms the linear baseline (+0.53%). This indicates that preserving some parallel memory via the IMB clipping mechanism is competitive, though for CIFAR-10 the strict orthogonality is a strong prior.

3. **Learnable IMB-Res Trade-offs**: The learnable budget variant (`005`) achieved **89.20%**, dipping slightly below the linear baseline. This suggests that allowing the model to freely learn the $\tau$ and $\kappa$ parameters without heavy regularization or constraint might lead to suboptimal radial retention on CIFAR-10, possibly overfitting the budget parameters.

4. **CIFAR-100 Benchmark**: The IMB-Res Fixed Budget run on CIFAR-100 (`006`) sets a strong marker with a top-1 accuracy of **71.97%** and a top-5 accuracy of **92.22%**. To fully contextualize this, linear and ORU baselines need to be run on CIFAR-100 for comparison, as suggested in the broader experiment matrix.

## Next Steps
- Run the Linear and ORU baselines on **CIFAR-100** to benchmark the IMB-Res 71.97% performance.
- Evaluate the IMB-Res fixed and learned variants over multiple seeds to measure variance.
- Investigate training dynamics (e.g., clipping rates) to understand why the learnable budget variant underperformed on CIFAR-10.
