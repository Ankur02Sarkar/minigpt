"""Tests for TrainingConfig / load_train_config (Phase 4.8 additions)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")

from training.config import TrainingConfig, load_train_config  # noqa: E402


def _write(path: Path, training: dict) -> None:
    cfg = {
        "model": {
            "vocab_size": 64,
            "num_layers": 1,
            "hidden_size": 32,
            "num_heads": 4,
            "max_position_embeddings": 32,
        },
        "training": training,
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_eval_max_batches_defaults_none(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, {"train_shards": ["tinystories.bin"], "val_shard": "val.bin"})
    _, tc = load_train_config(p)
    assert tc.eval_max_batches is None


def test_eval_max_batches_read_from_yaml(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(p, {"train_shards": ["tinystories.bin"], "val_shard": "val.bin", "eval_max_batches": 50})
    _, tc = load_train_config(p)
    assert tc.eval_max_batches == 50


def test_eval_max_batches_explicit_null(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write(
        p,
        {"train_shards": ["tinystories.bin"], "val_shard": "val.bin", "eval_max_batches": None},
    )
    _, tc = load_train_config(p)
    assert tc.eval_max_batches is None


def test_training_config_default_effective_batch() -> None:
    tc = TrainingConfig(
        train_shards=["tinystories.bin"], val_shard="val.bin", eval_max_batches=None
    )
    assert tc.effective_batch == tc.per_device_batch * tc.grad_accum
    assert tc.eval_max_batches is None
