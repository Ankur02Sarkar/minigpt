"""AdamW with decay / no-decay param groups."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.optim import AdamW

__all__ = ["build_optimizer"]


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.1,
    beta1: float = 0.9,
    beta2: float = 0.95,
) -> AdamW:
    """AdamW with no weight decay on biases and norm scales."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=(beta1, beta2), eps=1e-8)
