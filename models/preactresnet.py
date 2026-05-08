"""preactresnet in pytorch
https://github.com/weiaicunzai/pytorch-cifar100

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun

    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from connect import ConnLoggerMixin
from .gradient_checkpointing import Unsloth_Offloaded_Gradient_Checkpointer

class PreActBasic(ConnLoggerMixin, nn.Module):

    expansion = 1
    def __init__(
        self,
        in_channels, out_channels, stride, 
        gradient_checkpointing="none",
        log_interval=50, log_activations=True, 
        residual_connection="identity", orthogonal_method="global",
        residual_eps=1e-6, residual_perturbation=None,
        residual_pattern="default", residual_rescale_mode="scalar",
        # Neural SDE (radial) options (used when residual_connection="sde"/"chi"/"radial_sde")
        sde_alpha: float = 0.0,
        sde_beta: float | None = None,
        sde_sigma2: float = 1.0,
        sde_trainable: bool = False,
        sde_alpha_init: float = 1e-3,
        sde_beta_scale_init: float = 0.0,
        sde_noise_mode: str = "train",  # "train" | "always" | "off"
        imb_tau: float = 0.0,
        imb_kappa: float = 0.0,
        imb_trainable: bool = False,
        imb_tau_init: float = 0.0,
        imb_kappa_init: float = 0.0,
    ):
        nn.Module.__init__(self)
        ConnLoggerMixin.__init__(self,
                                 log_interval=log_interval,
                                 log_activations=log_activations)
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBasic.expansion, kernel_size=3, padding=1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBasic.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride)
        
        self.grad_ckpt = gradient_checkpointing
        if gradient_checkpointing:
            assert gradient_checkpointing in ("none", "torch", "unsloth"), "gradient_checkpointing should be one of 'none', 'torch', or 'unsloth'"
            self.grad_ckpt = gradient_checkpointing

        self.register_buffer("residual_eps", torch.tensor(residual_eps, dtype=torch.float32))
        pattern = (residual_pattern or "default").lower().replace("-", "_")
        rescale_mode = (residual_rescale_mode or "scalar").lower().replace("-", "_")
        self._res_kwargs = dict(method=residual_connection,
                                orthogonal_method=orthogonal_method,
                                perturbation=residual_perturbation,
                                pattern=pattern)
        if pattern == "rescale_stream":
            self._res_kwargs["rescale_mode"] = rescale_mode
        self._init_pattern_state(out_channels * PreActBasic.expansion, pattern, rescale_mode)
        self._init_sde_state(
            sde_alpha=sde_alpha,
            sde_beta=sde_beta,
            sde_sigma2=sde_sigma2,
            sde_trainable=sde_trainable,
            sde_alpha_init=sde_alpha_init,
            sde_beta_scale_init=sde_beta_scale_init,
            sde_noise_mode=sde_noise_mode,
        )
        self._init_imb_state(
            imb_tau=imb_tau,
            imb_kappa=imb_kappa,
            imb_trainable=imb_trainable,
            imb_tau_init=imb_tau_init,
            imb_kappa_init=imb_kappa_init,
        )
    def _forward_impl(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return self._connect_and_collect(shortcut, res, tag="conv", eps=self.residual_eps, **self._res_kwargs)

    def forward(self, x: torch.Tensor):
        if not self.training or self.grad_ckpt == "none":
            return self._forward_impl(x)

        elif self.grad_ckpt == "torch":
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)

        elif self.grad_ckpt == "unsloth":
            def _unsloth_fn(hidden_states):
                return (self._forward_impl(hidden_states),)
    
            return Unsloth_Offloaded_Gradient_Checkpointer.apply(
                _unsloth_fn, x
            )[0]

    def _init_pattern_state(self, channel_dim: int, pattern: str, rescale_mode: str) -> None:
        if pattern == "rezero":
            self._pattern_params["conv_alpha"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rezero_constrained":
            self._pattern_params["conv_theta"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rescale_stream":
            if rescale_mode == "conv1x1":
                proj = nn.Conv2d(channel_dim, channel_dim, kernel_size=1, bias=False)
                nn.init.zeros_(proj.weight)
                with torch.no_grad():
                    for c in range(channel_dim):
                        proj.weight[c, c, 0, 0] = 1.0
                self._pattern_modules["conv_rescale_proj"] = proj
            else:
                self._pattern_params["conv_rescale_alpha"] = nn.Parameter(torch.zeros(1))

    def _init_sde_state(
        self,
        *,
        sde_alpha: float,
        sde_beta: float | None,
        sde_sigma2: float,
        sde_trainable: bool,
        sde_alpha_init: float,
        sde_beta_scale_init: float,
        sde_noise_mode: str,
    ) -> None:
        # Fixed coefficients fallback (used when no trainable params are registered).
        self.sde_alpha = float(sde_alpha)
        self.sde_beta = None if sde_beta is None else float(sde_beta)
        self.sde_sigma2 = float(sde_sigma2)
        self.sde_noise_mode = str(sde_noise_mode)

        if not sde_trainable:
            return

        def inv_softplus(x: float) -> float:
            x = max(float(x), 1e-12)
            return math.log(math.expm1(x))

        raw_alpha_init = torch.tensor([inv_softplus(sde_alpha_init)], dtype=torch.float32)
        beta_scale_init = torch.tensor([float(sde_beta_scale_init)], dtype=torch.float32)
        self._pattern_params["conv_sde_raw_alpha"] = nn.Parameter(raw_alpha_init.clone())
        self._pattern_params["conv_sde_raw_beta_scale"] = nn.Parameter(beta_scale_init.clone())

    def _init_imb_state(
        self,
        *,
        imb_tau: float,
        imb_kappa: float,
        imb_trainable: bool,
        imb_tau_init: float,
        imb_kappa_init: float,
    ) -> None:
        self.imb_tau = float(imb_tau)
        self.imb_kappa = float(imb_kappa)

        if not imb_trainable:
            return

        def inv_softplus(x: float) -> float:
            x = max(float(x), 1e-12)
            return math.log(math.expm1(x))

        self._pattern_params["conv_imb_raw_tau"] = nn.Parameter(
            torch.tensor([inv_softplus(imb_tau_init)], dtype=torch.float32)
        )
        self._pattern_params["conv_imb_raw_kappa"] = nn.Parameter(
            torch.tensor([inv_softplus(imb_kappa_init)], dtype=torch.float32)
        )

class PreActBottleNeck(ConnLoggerMixin, nn.Module):

    expansion = 4
    def __init__(self,
        in_channels, out_channels, stride, 
        gradient_checkpointing="none",
        log_interval=50, log_activations=True, 
        residual_connection="identity", orthogonal_method="global",
        residual_eps=1e-6, residual_perturbation=None,
        residual_pattern="default", residual_rescale_mode="scalar",
        # Neural SDE (radial) options (used when residual_connection="sde"/"chi"/"radial_sde")
        sde_alpha: float = 0.0,
        sde_beta: float | None = None,
        sde_sigma2: float = 1.0,
        sde_trainable: bool = False,
        sde_alpha_init: float = 1e-3,
        sde_beta_scale_init: float = 0.0,
        sde_noise_mode: str = "train",  # "train" | "always" | "off"
        imb_tau: float = 0.0,
        imb_kappa: float = 0.0,
        imb_trainable: bool = False,
        imb_tau_init: float = 0.0,
        imb_kappa_init: float = 0.0,
    ):
        nn.Module.__init__(self)
        ConnLoggerMixin.__init__(self,
                                 log_interval=log_interval,
                                 log_activations=log_activations)

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, stride=stride),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * PreActBottleNeck.expansion, 1)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * PreActBottleNeck.expansion:
            self.shortcut = nn.Conv2d(in_channels, out_channels * PreActBottleNeck.expansion, 1, stride=stride)

        self.grad_ckpt = gradient_checkpointing
        if gradient_checkpointing:
            assert gradient_checkpointing in ("none", "torch", "unsloth"), "gradient_checkpointing should be one of 'none', 'torch', or 'unsloth'"
            self.grad_ckpt = gradient_checkpointing

        self.register_buffer("residual_eps", torch.tensor(residual_eps, dtype=torch.float32))
        pattern = (residual_pattern or "default").lower().replace("-", "_")
        rescale_mode = (residual_rescale_mode or "scalar").lower().replace("-", "_")
        self._res_kwargs = dict(method=residual_connection,
                                orthogonal_method=orthogonal_method,
                                perturbation=residual_perturbation,
                                pattern=pattern)
        if pattern == "rescale_stream":
            self._res_kwargs["rescale_mode"] = rescale_mode
        channels = out_channels * PreActBottleNeck.expansion
        self._init_pattern_state(channels, pattern, rescale_mode)
        self._init_sde_state(
            sde_alpha=sde_alpha,
            sde_beta=sde_beta,
            sde_sigma2=sde_sigma2,
            sde_trainable=sde_trainable,
            sde_alpha_init=sde_alpha_init,
            sde_beta_scale_init=sde_beta_scale_init,
            sde_noise_mode=sde_noise_mode,
        )
        self._init_imb_state(
            imb_tau=imb_tau,
            imb_kappa=imb_kappa,
            imb_trainable=imb_trainable,
            imb_tau_init=imb_tau_init,
            imb_kappa_init=imb_kappa_init,
        )
    
    def _forward_impl(self, x):

        res = self.residual(x)
        shortcut = self.shortcut(x)

        return self._connect_and_collect(shortcut, res, tag="conv", eps=self.residual_eps, **self._res_kwargs)

    def forward(self, x: torch.Tensor):
        if not self.training or self.grad_ckpt == "none":
            return self._forward_impl(x)

        elif self.grad_ckpt == "torch":
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)

        elif self.grad_ckpt == "unsloth":
            def _unsloth_fn(hidden_states):
                return (self._forward_impl(hidden_states),)   # ← 튜플 반환
    
            return Unsloth_Offloaded_Gradient_Checkpointer.apply(
                _unsloth_fn, x
            )[0]

    def _init_pattern_state(self, channel_dim: int, pattern: str, rescale_mode: str) -> None:
        if pattern == "rezero":
            self._pattern_params["conv_alpha"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rezero_constrained":
            self._pattern_params["conv_theta"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rescale_stream":
            if rescale_mode == "conv1x1":
                proj = nn.Conv2d(channel_dim, channel_dim, kernel_size=1, bias=False)
                nn.init.zeros_(proj.weight)
                with torch.no_grad():
                    for c in range(channel_dim):
                        proj.weight[c, c, 0, 0] = 1.0
                self._pattern_modules["conv_rescale_proj"] = proj
            else:
                self._pattern_params["conv_rescale_alpha"] = nn.Parameter(torch.zeros(1))

    def _init_sde_state(
        self,
        *,
        sde_alpha: float,
        sde_beta: float | None,
        sde_sigma2: float,
        sde_trainable: bool,
        sde_alpha_init: float,
        sde_beta_scale_init: float,
        sde_noise_mode: str,
    ) -> None:
        # Fixed coefficients fallback (used when no trainable params are registered).
        self.sde_alpha = float(sde_alpha)
        self.sde_beta = None if sde_beta is None else float(sde_beta)
        self.sde_sigma2 = float(sde_sigma2)
        self.sde_noise_mode = str(sde_noise_mode)

        if not sde_trainable:
            return

        def inv_softplus(x: float) -> float:
            x = max(float(x), 1e-12)
            return math.log(math.expm1(x))

        raw_alpha_init = torch.tensor([inv_softplus(sde_alpha_init)], dtype=torch.float32)
        beta_scale_init = torch.tensor([float(sde_beta_scale_init)], dtype=torch.float32)
        self._pattern_params["conv_sde_raw_alpha"] = nn.Parameter(raw_alpha_init.clone())
        self._pattern_params["conv_sde_raw_beta_scale"] = nn.Parameter(beta_scale_init.clone())

    def _init_imb_state(
        self,
        *,
        imb_tau: float,
        imb_kappa: float,
        imb_trainable: bool,
        imb_tau_init: float,
        imb_kappa_init: float,
    ) -> None:
        self.imb_tau = float(imb_tau)
        self.imb_kappa = float(imb_kappa)

        if not imb_trainable:
            return

        def inv_softplus(x: float) -> float:
            x = max(float(x), 1e-12)
            return math.log(math.expm1(x))

        self._pattern_params["conv_imb_raw_tau"] = nn.Parameter(
            torch.tensor([inv_softplus(imb_tau_init)], dtype=torch.float32)
        )
        self._pattern_params["conv_imb_raw_kappa"] = nn.Parameter(
            torch.tensor([inv_softplus(imb_kappa_init)], dtype=torch.float32)
        )

class PreActResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100, **kwargs):
        super().__init__()
        self.input_channels = 64

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        print("connection method:", kwargs.get("residual_connection", None), "orthogonal", kwargs.get("orthogonal_method", None))

        is_layernorm_classifier = kwargs.pop("is_layernorm_classifier", False)
        

        self.stage1 = self._make_layers(block, num_block[0], 64,  1, **kwargs)
        self.stage2 = self._make_layers(block, num_block[1], 128, 2, **kwargs)
        self.stage3 = self._make_layers(block, num_block[2], 256, 2, **kwargs)
        self.stage4 = self._make_layers(block, num_block[3], 512, 2, **kwargs)

        if is_layernorm_classifier:
            self.linear = nn.Sequential(
                nn.LayerNorm(self.input_channels),
                nn.Linear(self.input_channels, num_classes)
            )
        else:
            self.linear = nn.Linear(self.input_channels, num_classes)
        # self.linear = nn.Linear(self.input_channels, num_classes)

    def _make_layers(self, block, block_num, out_channels, stride, **kwargs):
        layers = []

        layers.append(block(self.input_channels, out_channels, stride, **kwargs))
        self.input_channels = out_channels * block.expansion

        while block_num - 1:
            layers.append(block(self.input_channels, out_channels, 1, **kwargs))
            self.input_channels = out_channels * block.expansion
            block_num -= 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def pop_stats(self, *, scalarize: bool = True):
        all_stats = []
        for module in self.modules():
            if isinstance(module, ConnLoggerMixin):
                all_stats.extend(module.pop_stats(scalarize=scalarize))
        return all_stats


def preactresnet18(**kwargs):
    return PreActResNet(PreActBasic, [2, 2, 2, 2], **kwargs)

def preactresnet34(**kwargs):
    return PreActResNet(PreActBasic, [3, 4, 6, 3], **kwargs)

def preactresnet50(**kwargs):
    return PreActResNet(PreActBottleNeck, [3, 4, 6, 3], **kwargs)

def preactresnet101(**kwargs):
    return PreActResNet(PreActBottleNeck, [3, 4, 23, 3], **kwargs)

def preactresnet152(**kwargs):
    return PreActResNet(PreActBottleNeck, [3, 8, 36, 3], **kwargs)

PRESET_PREACT_RESNET = {
    "resnet18": preactresnet18,
    "resnet34": preactresnet34,
    "resnet50": preactresnet50,
    "resnet101": preactresnet101,
    "resnet152": preactresnet152
}
