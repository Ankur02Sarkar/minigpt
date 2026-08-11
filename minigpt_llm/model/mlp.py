"""SwiGLU feed-forward network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt_llm.model.config import ModelConfig, swiglu_hidden_dim

__all__ = ["SwiGLU", "swiglu_hidden_dim"]


class SwiGLU(nn.Module):
    """SwiGLU MLP: ``down(silu(gate(x)) * up(x))`` with no biases."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        h = config.hidden_size
        i = config.intermediate_size
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
