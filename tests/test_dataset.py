"""Tests for memmap TokenShardDataset and smoke batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.dataset import TokenShardDataset, build_batch, main  # noqa: E402


def _write_shard(path: Path, n: int = 5000) -> None:
    arr = np.arange(n, dtype=np.int32)
    arr.tofile(path)


def test_build_batch_shapes_and_shift(tmp_path: Path) -> None:
    shard = tmp_path / "t.bin"
    _write_shard(shard, 5000)
    batch = build_batch(shard, context=128, batch=4, seed=0)
    assert batch["input_ids"].shape == (4, 128)
    assert batch["labels"].shape == (4, 128)
    assert torch.equal(batch["labels"][:, :-1], batch["input_ids"][:, 1:])


def test_dataset_too_short(tmp_path: Path) -> None:
    shard = tmp_path / "t.bin"
    _write_shard(shard, 10)
    with pytest.raises(ValueError, match="need at least"):
        TokenShardDataset(shard, context_length=32)


def test_cli_smoke(tmp_path: Path) -> None:
    shard = tmp_path / "t.bin"
    _write_shard(shard, 5000)
    rc = main(["--shard", str(shard), "--context", "64", "--batch", "2"])
    assert rc == 0


def test_cli_missing_shard(tmp_path: Path) -> None:
    rc = main(["--shard", str(tmp_path / "missing.bin"), "--context", "8", "--batch", "1"])
    assert rc == 1
