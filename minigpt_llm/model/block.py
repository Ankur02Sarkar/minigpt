"""RMSNorm and pre-norm transformer decoder block."""

from __future__ import annotations

import torch
import torch.nn as nn

from minigpt_llm.model.attention import CausalSelfAttention, KVCache
from minigpt_llm.model.config import ModelConfig
from minigpt_llm.model.mlp import SwiGLU

__all__ = ["DecoderBlock", "RMSNorm"]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean centering)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for stability
        orig_dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(var + self.eps)
        return (x_norm * self.weight.float()).to(orig_dtype)


class DecoderBlock(nn.Module):
    """Pre-norm attention + residual, pre-norm MLP + residual."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size)
        self.attn = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache | None]:
        h, present = self.attn(
            self.attn_norm(x),
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        x = x + h
        x = x + self.mlp(self.mlp_norm(x))
        return x, present
