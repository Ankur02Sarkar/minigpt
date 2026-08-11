"""Linear warmup → cosine decay LR schedule (pure PyTorch)."""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer

__all__ = ["WarmupCosineScheduler"]


class WarmupCosineScheduler:
    """Step-based LR schedule with state_dict for checkpointing.

    - Steps ``[0, warmup_steps)``: linear warmup from 0 → peak_lr
    - Steps ``[warmup_steps, max_steps]``: cosine decay to ``peak_lr * min_lr_ratio``
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        peak_lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
        last_step: int = -1,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.last_step = last_step
        # set initial LR for step 0 preview
        self._set_lr(self.get_lr(0))

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            if self.warmup_steps == 0:
                return self.peak_lr
            return self.peak_lr * float(step + 1) / float(self.warmup_steps)
        if step >= self.max_steps:
            return self.peak_lr * self.min_lr_ratio
        progress = float(step - self.warmup_steps) / float(
            max(1, self.max_steps - self.warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr = self.peak_lr * self.min_lr_ratio
        return min_lr + (self.peak_lr - min_lr) * cosine

    def step(self) -> float:
        self.last_step += 1
        lr = self.get_lr(self.last_step)
        self._set_lr(lr)
        return lr

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def state_dict(self) -> dict[str, Any]:
        return {
            "last_step": self.last_step,
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.last_step = int(state["last_step"])
        self.peak_lr = float(state["peak_lr"])
        self.warmup_steps = int(state["warmup_steps"])
        self.max_steps = int(state["max_steps"])
        self.min_lr_ratio = float(state["min_lr_ratio"])
        self._set_lr(self.get_lr(max(0, self.last_step)))
