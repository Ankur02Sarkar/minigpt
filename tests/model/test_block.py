"""Tests for decoder block residual stream."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.block import DecoderBlock, RMSNorm
from minigpt_llm.model.config import ModelConfig


def test_rmsnorm_unit_scale() -> None:
    norm = RMSNorm(32)
    x = torch.randn(2, 4, 32) * 5
    y = norm(x)
    # RMS along last dim ≈ 1 (weight init ones)
    rms = y.float().pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), rtol=1e-4, atol=1e-4)


def test_block_residual_bounded() -> None:
    cfg = ModelConfig(
        vocab_size=100,
        num_layers=1,
        hidden_size=64,
        num_heads=4,
        max_position_embeddings=32,
        dropout=0.0,
    )
    block = DecoderBlock(cfg)
    block.eval()
    x = torch.randn(2, 8, 64)
    y, _ = block(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    # Residual stream should not explode from single layer
    assert y.norm().item() < x.norm().item() * 50 + 100
