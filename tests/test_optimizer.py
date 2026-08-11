"""Tests for AdamW param groups."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.config import ModelConfig
from minigpt_llm.model.model import GPT
from training.optimizer import build_optimizer


def test_decay_groups_split() -> None:
    cfg = ModelConfig(
        vocab_size=64,
        num_layers=1,
        hidden_size=32,
        num_heads=4,
        max_position_embeddings=32,
    )
    model = GPT(cfg)
    opt = build_optimizer(model, lr=1e-3, weight_decay=0.1)
    assert len(opt.param_groups) == 2
    decay_g, nodecay_g = opt.param_groups
    assert decay_g["weight_decay"] == 0.1
    assert nodecay_g["weight_decay"] == 0.0
    n_decay = sum(p.numel() for p in decay_g["params"])
    n_nodecay = sum(p.numel() for p in nodecay_g["params"])
    assert n_decay > 0
    assert n_nodecay > 0
    assert n_decay + n_nodecay == model.num_parameters()
