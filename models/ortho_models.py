import torch
import torch.nn as nn
import math
from timm.layers import DropPath, Mlp
from timm.models.vision_transformer import Attention
from connect import ConnLoggerMixin
from .gradient_checkpointing import Unsloth_Offloaded_Gradient_Checkpointer


class OrthoBlock(ConnLoggerMixin, nn.Module):
    """
    OrthoBlock with orthogonal residual connection
    """
    _global_id = 0
    def __init__(
        self, 
        hidden_size, 
        num_heads,
        mlp_ratio=4.0,
    residual_connection="orthogonal",
    orthogonal_method="global",
        residual_eps=1e-6,
        residual_perturbation=None,
    residual_pattern="default",
    residual_rescale_mode="scalar",
        # Neural SDE (radial) options (used when residual_connection="sde"/"chi"/"radial_sde")
        sde_alpha: float = 0.0,
        sde_beta: float | None = None,
        sde_sigma2: float = 1.0,
        sde_trainable: bool = False,
        sde_alpha_init: float = 1e-3,
        sde_beta_scale_init: float = 0.0,
        sde_noise_mode: str = "train",  # "train" | "always" | "off"
        modulate=True,
        mlp_dropout=0.0,
        drop_path=0.0,
        log_interval=50,
        log_activations=True,
        gradient_checkpointing="none", # "none", "torch", "unsloth"
        **block_kwargs
    ):
        nn.Module.__init__(self)
        ConnLoggerMixin.__init__(self, log_interval=log_interval,
                                 log_activations=log_activations)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.attn.fused_attn = True
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=mlp_dropout)

        self.drop_path = (
            nn.Identity() if drop_path == 0.0 else
            DropPath(drop_path)
        )
        self.register_buffer(
            "residual_eps",
            torch.tensor([residual_eps], dtype=torch.float32)
        )

        pattern = (residual_pattern or "default").lower().replace("-", "_")
        rescale_mode = (residual_rescale_mode or "scalar").lower().replace("-", "_")
        self.residual_kwargs = {
            "method": residual_connection,
            "orthogonal_method": orthogonal_method,
            "perturbation": residual_perturbation,
            "pattern": pattern,
        }
        if pattern == "rescale_stream":
            self.residual_kwargs["rescale_mode"] = rescale_mode
        if gradient_checkpointing:
            assert gradient_checkpointing in ("none", "torch", "unsloth"), "gradient_checkpointing should be one of 'none', 'torch', or 'unsloth'"
            self.grad_ckpt = gradient_checkpointing

        self._init_pattern_state(hidden_size, pattern, rescale_mode)
        self._init_sde_state(
            sde_alpha=sde_alpha,
            sde_beta=sde_beta,
            sde_sigma2=sde_sigma2,
            sde_trainable=sde_trainable,
            sde_alpha_init=sde_alpha_init,
            sde_beta_scale_init=sde_beta_scale_init,
            sde_noise_mode=sde_noise_mode,
        )


    def set_step_fn(self, fn):      # 한 번만 호출
        self._get_step = fn

    def perturbation(self):
        return None
    
    def forward(self, x: torch.Tensor):
        if not self.training or self.grad_ckpt == "none":
            return self._forward_impl(x)

        # torch.utils.checkpoint 
        elif self.grad_ckpt == "torch":
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)

        elif self.grad_ckpt == "unsloth":
            def _unsloth_fn(hidden_states):
                return (self._forward_impl(hidden_states),)

            return Unsloth_Offloaded_Gradient_Checkpointer.apply(
                _unsloth_fn, x
            )[0]

    def _forward_impl(self, x: torch.Tensor):
        attn_out = self.drop_path(self.attn(self.norm1(x)))
        x = self._connect_and_collect(
            x, attn_out, 
            tag="attn", eps=self.residual_eps, **self.residual_kwargs
        )
        mlp_out = self.drop_path(self.mlp(self.norm2(x)))
        x = self._connect_and_collect(
            x, mlp_out, 
            tag="mlp", eps=self.residual_eps, **self.residual_kwargs
        )
        return x

    def _init_pattern_state(self, hidden_size: int, pattern: str, rescale_mode: str) -> None:
        if pattern == "rezero":
            self._pattern_params["attn_alpha"] = nn.Parameter(torch.zeros(1))
            self._pattern_params["mlp_alpha"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rezero_constrained":
            self._pattern_params["attn_theta"] = nn.Parameter(torch.zeros(1))
            self._pattern_params["mlp_theta"] = nn.Parameter(torch.zeros(1))
        elif pattern == "rescale_stream":
            if rescale_mode != "scalar":
                raise ValueError("residual_rescale_mode='conv1x1' is not supported for ViT blocks")
            self._pattern_params["attn_rescale_alpha"] = nn.Parameter(torch.zeros(1))
            self._pattern_params["mlp_rescale_alpha"] = nn.Parameter(torch.zeros(1))

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
        self._pattern_params["attn_sde_raw_alpha"] = nn.Parameter(raw_alpha_init.clone())
        self._pattern_params["mlp_sde_raw_alpha"] = nn.Parameter(raw_alpha_init.clone())
        self._pattern_params["attn_sde_raw_beta_scale"] = nn.Parameter(beta_scale_init.clone())
        self._pattern_params["mlp_sde_raw_beta_scale"] = nn.Parameter(beta_scale_init.clone())
