"""Tests for ModelConfig, YAML load, param estimate."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.config import ModelConfig, estimate_params, load_config, swiglu_hidden_dim
from minigpt_llm.model.model import GPT


def test_swiglu_hidden_dim_multiple_of_64() -> None:
    for h in (256, 384, 512, 768):
        i = swiglu_hidden_dim(h)
        assert i % 64 == 0
        assert i >= int(8 * h / 3)


def test_model_config_head_dim() -> None:
    c = ModelConfig(
        vocab_size=32000,
        num_layers=2,
        hidden_size=256,
        num_heads=4,
        max_position_embeddings=128,
    )
    assert c.head_dim == 64


def test_invalid_heads() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(
            vocab_size=100,
            num_layers=1,
            hidden_size=100,
            num_heads=3,
            max_position_embeddings=32,
        )


def test_load_minigpt_low_yaml() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "minigpt-low.yaml")
    assert cfg.vocab_size == 32000
    assert cfg.num_layers == 6
    assert cfg.hidden_size == 256
    assert cfg.num_heads == 4
    assert cfg.max_position_embeddings == 512
    assert cfg.tie_weights is True


def test_load_minigpt_high_yaml() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "minigpt-high.yaml")
    assert cfg.num_layers == 8
    assert cfg.hidden_size == 384
    assert cfg.num_heads == 6
    assert cfg.max_position_embeddings == 1024


def test_estimate_params_within_1_percent_low() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "minigpt-low.yaml")
    est = estimate_params(cfg)
    model = GPT(cfg)
    real = model.num_parameters()
    rel = abs(real - est) / max(real, 1)
    assert rel < 0.01, f"estimate {est} vs real {real} rel={rel}"
    assert 4_000_000 < real < 15_000_000


def test_estimate_params_high_about_26m() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / "configs" / "minigpt-high.yaml")
    model = GPT(cfg)
    real = model.num_parameters()
    est = estimate_params(cfg)
    rel = abs(real - est) / max(real, 1)
    assert rel < 0.01
    assert 15_000_000 < real < 30_000_000
