"""Tests for full GPT forward and generate."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.config import ModelConfig
from minigpt_llm.model.init import assert_not_bfloat16
from minigpt_llm.model.model import GPT


def _small_cfg() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        max_position_embeddings=64,
        dropout=0.0,
        tie_weights=True,
    )


def test_forward_logits_and_loss() -> None:
    model = GPT(_small_cfg())
    model.train()
    ids = torch.randint(0, 128, (2, 16))
    out = model(ids, labels=ids)
    assert out.logits.shape == (2, 16, 128)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()


def test_tied_weights() -> None:
    model = GPT(_small_cfg())
    assert model.lm_head.weight is model.embed_tokens.weight


def test_generate_grows_sequence() -> None:
    model = GPT(_small_cfg())
    model.eval()
    ids = torch.randint(0, 128, (1, 5))
    out = model.generate(ids, max_new_tokens=7, temperature=0.0)
    assert out.shape == (1, 12)


def test_generate_with_temperature() -> None:
    model = GPT(_small_cfg())
    model.eval()
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (1, 3))
    out = model.generate(ids, max_new_tokens=4, temperature=0.8, top_k=10)
    assert out.shape[1] == 7


def test_reject_bfloat16() -> None:
    with pytest.raises(ValueError, match="bfloat16"):
        assert_not_bfloat16(torch.bfloat16)
