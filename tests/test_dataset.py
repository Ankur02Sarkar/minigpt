"""Tests for memmap TokenShardDataset and smoke batch."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.dataset import (  # noqa: E402
    MultiShardDataset,
    TokenShardDataset,
    _reseed_worker_dataset,
    build_batch,
    main,
)


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


def test_multi_shard_samples_token_weighted(tmp_path: Path) -> None:
    """Shard selection must be proportional to token count, not uniform 1/N."""
    big = tmp_path / "big.bin"  # 4000 tokens, values 0..3999
    small = tmp_path / "small.bin"  # 1000 tokens, values 10000..10999
    np.arange(0, 4000, dtype=np.int32).tofile(big)
    np.arange(10000, 11000, dtype=np.int32).tofile(small)

    ds = MultiShardDataset([big, small], context_length=16, seed=0)
    # weights: big=0.8, small=0.2
    assert pytest.approx(ds._shard_probs[0], rel=1e-6) == 0.8
    assert pytest.approx(ds._shard_probs[1], rel=1e-6) == 0.2

    n = 800
    big_hits = sum(int(ds[i]["input_ids"][0].item()) < 10000 for i in range(n))
    frac = big_hits / n
    # expected 0.80; generous tolerance for sampling noise (<3 std)
    assert abs(frac - 0.8) < 0.08, f"big shard fraction {frac} far from 0.8"


def test_reseed_breaks_worker_overlap(tmp_path: Path) -> None:
    """Two dataset copies with the same seed draw identical windows (the bug);
    re-seeding each with a distinct worker seed breaks the collision."""
    shard = tmp_path / "t.bin"
    _write_shard(shard, 5000)
    ds0 = TokenShardDataset(shard, context_length=128, seed=42)
    ds1 = TokenShardDataset(shard, context_length=128, seed=42)
    first = [int(ds0[i]["input_ids"][0].item()) for i in range(24)]
    same = [int(ds1[i]["input_ids"][0].item()) for i in range(24)]
    assert first == same  # pre-fix: forked workers draw the same windows

    _reseed_worker_dataset(ds0, 100)
    _reseed_worker_dataset(ds1, 101)
    again0 = [int(ds0[i]["input_ids"][0].item()) for i in range(24)]
    again1 = [int(ds1[i]["input_ids"][0].item()) for i in range(24)]
    assert again0 != again1  # post-fix: workers diverge
