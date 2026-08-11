"""Weight initialization and dtype policy helpers."""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["INIT_STD", "apply_weight_init", "assert_not_bfloat16"]

INIT_STD = 0.02


def assert_not_bfloat16(dtype: torch.dtype) -> None:
    """T4 (Turing) has no BF16 — refuse explicitly."""
    if dtype == torch.bfloat16:
        raise ValueError("torch.bfloat16 is not supported on Tesla T4 (Turing). Use torch.float16.")


def apply_weight_init(module: nn.Module) -> None:
    """Truncated-normal linears (std 0.02); normal embeddings; ones/zeros for RMSNorm."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=INIT_STD, a=-2 * INIT_STD, b=2 * INIT_STD)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=INIT_STD)
