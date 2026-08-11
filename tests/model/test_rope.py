"""Tests for RoPE."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.rope import RotaryEmbedding, apply_rotary_emb


def test_apply_rotary_shape() -> None:
    b, h, t, d = 2, 4, 8, 16
    x = torch.randn(b, h, t, d)
    rope = RotaryEmbedding(head_dim=d, max_seq_len=32)
    cos, sin = rope(t)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    y = apply_rotary_emb(x, cos, sin)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_rope_odd_head_dim_rejected() -> None:
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(head_dim=15, max_seq_len=8)


def test_rope_preserves_norm_approximately() -> None:
    """RoPE is an orthogonal transform on pairs — L2 per head dim ≈ preserved."""
    t, d = 16, 32
    x = torch.randn(1, 1, t, d)
    rope = RotaryEmbedding(head_dim=d, max_seq_len=64)
    cos, sin = rope(t)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    y = apply_rotary_emb(x, cos, sin)
    xn = x.norm(dim=-1)
    yn = y.norm(dim=-1)
    assert torch.allclose(xn, yn, rtol=1e-4, atol=1e-5)


def test_rope_offset_matches_slice() -> None:
    rope = RotaryEmbedding(head_dim=16, max_seq_len=64)
    cos_full, sin_full = rope(10, offset=0)
    cos_off, sin_off = rope(5, offset=5)
    assert torch.allclose(cos_full[5:10], cos_off)
    assert torch.allclose(sin_full[5:10], sin_off)
