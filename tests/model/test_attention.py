"""Tests for causal self-attention."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.attention import CausalSelfAttention
from minigpt_llm.model.config import ModelConfig


def _cfg(**kwargs: int | float) -> ModelConfig:
    base = dict(
        vocab_size=100,
        num_layers=1,
        hidden_size=64,
        num_heads=4,
        max_position_embeddings=64,
        dropout=0.0,
    )
    base.update(kwargs)
    return ModelConfig(**base)  # type: ignore[arg-type]


def test_attention_output_shape() -> None:
    attn = CausalSelfAttention(_cfg())
    attn.eval()
    x = torch.randn(2, 8, 64)
    y, present = attn(x, use_cache=False)
    assert y.shape == (2, 8, 64)
    assert present is None
    assert torch.isfinite(y).all()


def test_attention_cache_grows() -> None:
    attn = CausalSelfAttention(_cfg())
    attn.eval()
    x1 = torch.randn(1, 4, 64)
    y1, past = attn(x1, use_cache=True)
    assert past is not None
    assert past[0].shape[2] == 4
    x2 = torch.randn(1, 1, 64)
    y2, past2 = attn(x2, past_key_value=past, use_cache=True)
    assert y2.shape == (1, 1, 64)
    assert past2 is not None
    assert past2[0].shape[2] == 5


def test_causal_mask_lower_triangular_property() -> None:
    """Changing a future token must not change past outputs (train path)."""
    attn = CausalSelfAttention(_cfg())
    attn.train()
    torch.manual_seed(0)
    x = torch.randn(1, 6, 64)
    y1, _ = attn(x)

    x2 = x.clone()
    x2[0, -1] = torch.randn(64)  # alter last position only
    y2, _ = attn(x2)
    # Positions before last should match (causal)
    assert torch.allclose(y1[0, :-1], y2[0, :-1], rtol=1e-4, atol=1e-5)
