"""Training hyperparameters loaded from YAML (alongside model config)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from minigpt_llm.model.config import ModelConfig, load_config

__all__ = ["TrainingConfig", "load_train_config"]


@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop (CLI may override fields)."""

    seed: int = 42
    context_length: int = 512
    per_device_batch: int = 4
    grad_accum: int = 16
    max_steps: int = 50_000
    lr: float = 5.0e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_steps: int = 2000
    min_lr_ratio: float = 0.1
    eval_every: int = 1000
    ckpt_every: int = 1000
    log_every: int = 10
    grad_clip: float = 1.0
    num_workers: int = 2
    train_shards: list[str] = field(default_factory=lambda: ["tinystories.bin"])
    val_shard: str = "val.bin"

    def __post_init__(self) -> None:
        if self.context_length < 1:
            raise ValueError("context_length must be >= 1")
        if self.per_device_batch < 1:
            raise ValueError("per_device_batch must be >= 1")
        if self.grad_accum < 1:
            raise ValueError("grad_accum must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not self.train_shards:
            raise ValueError("train_shards must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def effective_batch(self) -> int:
        return self.per_device_batch * self.grad_accum


def load_train_config(path: Path | str) -> tuple[ModelConfig, TrainingConfig]:
    """Load model + training config from one YAML file."""
    p = Path(path)
    model_cfg = load_config(p)
    with p.open("r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError("config root must be a mapping")
    t_raw = raw.get("training", {})
    if t_raw is None:
        t_raw = {}
    if not isinstance(t_raw, dict):
        raise TypeError("training section must be a mapping")

    train_cfg = TrainingConfig(
        seed=int(t_raw.get("seed", 42)),
        context_length=int(t_raw.get("context_length", model_cfg.max_position_embeddings)),
        per_device_batch=int(t_raw.get("per_device_batch", 4)),
        grad_accum=int(t_raw.get("grad_accum", 16)),
        max_steps=int(t_raw.get("max_steps", 50_000)),
        lr=float(t_raw.get("lr", 5.0e-4)),
        weight_decay=float(t_raw.get("weight_decay", 0.1)),
        beta1=float(t_raw.get("beta1", 0.9)),
        beta2=float(t_raw.get("beta2", 0.95)),
        warmup_steps=int(t_raw.get("warmup_steps", 2000)),
        min_lr_ratio=float(t_raw.get("min_lr_ratio", 0.1)),
        eval_every=int(t_raw.get("eval_every", 1000)),
        ckpt_every=int(t_raw.get("ckpt_every", 1000)),
        log_every=int(t_raw.get("log_every", 10)),
        grad_clip=float(t_raw.get("grad_clip", 1.0)),
        num_workers=int(t_raw.get("num_workers", 2)),
        train_shards=list(t_raw.get("train_shards", ["tinystories.bin"])),
        val_shard=str(t_raw.get("val_shard", "val.bin")),
    )
    if train_cfg.context_length > model_cfg.max_position_embeddings:
        raise ValueError(
            f"training.context_length ({train_cfg.context_length}) exceeds "
            f"model.max_position_embeddings ({model_cfg.max_position_embeddings})"
        )
    return model_cfg, train_cfg
