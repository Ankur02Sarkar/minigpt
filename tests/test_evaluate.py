"""Tests for validation evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.config import ModelConfig
from minigpt_llm.model.model import GPT
from training.evaluate import evaluate_shard


def test_evaluate_shard(tmp_path: Path) -> None:
    cfg = ModelConfig(
        vocab_size=64,
        num_layers=1,
        hidden_size=32,
        num_heads=4,
        max_position_embeddings=32,
        dropout=0.0,
    )
    model = GPT(cfg)
    tokens = np.random.randint(0, 64, size=500, dtype=np.int32)
    shard = tmp_path / "val.bin"
    tokens.tofile(shard)
    metrics = evaluate_shard(
        model, shard, context_length=16, batch_size=4, max_batches=5, device="cpu"
    )
    assert "val_loss" in metrics and "val_ppl" in metrics
    assert metrics["val_loss"] > 0
    assert metrics["val_ppl"] > 1.0
