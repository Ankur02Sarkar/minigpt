"""Model configuration: frozen dataclass, YAML loader, param estimate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ModelConfig", "estimate_params", "load_config", "swiglu_hidden_dim"]


def swiglu_hidden_dim(hidden_size: int) -> int:
    """Intermediate size for SwiGLU: ≈ 8/3 * H, rounded up to multiple of 64."""
    raw = int(8 * hidden_size / 3)
    return ((raw + 63) // 64) * 64


@dataclass(frozen=True)
class ModelConfig:
    """GPT architecture hyperparameters."""

    vocab_size: int
    num_layers: int
    hidden_size: int
    num_heads: int
    max_position_embeddings: int
    dropout: float = 0.1
    rope_theta: float = 10_000.0
    tie_weights: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be > 0, got {self.hidden_size}")
        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {self.num_heads}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be > 0, got {self.max_position_embeddings}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be > 0, got {self.rope_theta}")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def intermediate_size(self) -> int:
        return swiglu_hidden_dim(self.hidden_size)


def load_config(path: Path | str) -> ModelConfig:
    """Load ``ModelConfig`` from a YAML file.

    Accepts either nested ``model:`` keys (preferred) or a flat mapping of fields.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw: Any = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping, got {type(raw)}")
    data = raw.get("model", raw)
    if not isinstance(data, dict):
        raise TypeError(f"model section must be a mapping, got {type(data)}")
    required = {
        "vocab_size",
        "num_layers",
        "hidden_size",
        "num_heads",
        "max_position_embeddings",
    }
    missing = required - set(data)
    if missing:
        raise KeyError(f"config missing required keys: {sorted(missing)}")
    return ModelConfig(
        vocab_size=int(data["vocab_size"]),
        num_layers=int(data["num_layers"]),
        hidden_size=int(data["hidden_size"]),
        num_heads=int(data["num_heads"]),
        max_position_embeddings=int(data["max_position_embeddings"]),
        dropout=float(data.get("dropout", 0.1)),
        rope_theta=float(data.get("rope_theta", 10_000.0)),
        tie_weights=bool(data.get("tie_weights", True)),
    )


def estimate_params(config: ModelConfig) -> int:
    """Estimate total trainable parameters (matches no-bias LLaMA-style layout).

    - Token embed: ``V * H`` (LM head tied → not double-counted when ``tie_weights``)
    - Per layer: MHA ``4 H²`` (q,k,v,o) + SwiGLU ``3 H * I`` + 2× RMSNorm ``2 H``
    - Final RMSNorm: ``H``
    - Untied LM head: ``V * H``
    """
    h = config.hidden_size
    v = config.vocab_size
    n = config.num_layers
    i = config.intermediate_size

    embed = v * h
    per_layer = 4 * h * h + 3 * h * i + 2 * h  # attn + mlp + 2 norms
    final_norm = h
    head = 0 if config.tie_weights else v * h
    return embed + n * per_layer + final_norm + head
