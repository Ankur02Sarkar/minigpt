"""Tests for SwiGLU MLP."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.config import ModelConfig, swiglu_hidden_dim
from minigpt_llm.model.mlp import SwiGLU


def test_swiglu_param_count() -> None:
    cfg = ModelConfig(
        vocab_size=100,
        num_layers=1,
        hidden_size=256,
        num_heads=4,
        max_position_embeddings=32,
    )
    mlp = SwiGLU(cfg)
    h, i = cfg.hidden_size, cfg.intermediate_size
    expected = 3 * h * i  # gate, up, down — no biases
    actual = sum(p.numel() for p in mlp.parameters())
    assert actual == expected
    assert i == swiglu_hidden_dim(h)


def test_swiglu_forward_shape() -> None:
    cfg = ModelConfig(
        vocab_size=50,
        num_layers=1,
        hidden_size=64,
        num_heads=4,
        max_position_embeddings=16,
    )
    mlp = SwiGLU(cfg)
    x = torch.randn(2, 5, 64)
    y = mlp(x)
    assert y.shape == x.shape
