"""Memory-mapped token shard dataset for causal LM training (Phase 1.9 smoke + Phase 3)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import structlog
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "MultiShardDataset",
    "TokenShardDataset",
    "build_batch",
    "build_dataloader",
    "infinite_loader",
    "main",
    "worker_init_fn",
]

log = structlog.get_logger(__name__)

_DTYPE = np.dtype("<i4")


class TokenShardDataset(Dataset[dict[str, torch.Tensor]]):
    """Sample random ``(context+1)`` windows from an int32 ``.bin`` shard.

    Each item returns::

        {"input_ids": LongTensor[T], "labels": LongTensor[T]}

    where ``labels`` is ``input_ids`` shifted by one (next-token prediction).
    """

    def __init__(
        self,
        shard_path: Path | str,
        *,
        context_length: int = 1024,
        seed: int = 42,
        num_samples: int | None = None,
    ) -> None:
        self.shard_path = Path(shard_path)
        if not self.shard_path.is_file():
            raise FileNotFoundError(f"shard not found: {self.shard_path}")
        if context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length}")

        self.context_length = context_length
        self.data = np.memmap(self.shard_path, dtype=_DTYPE, mode="r")
        self.n_tokens = int(self.data.shape[0])
        window = context_length + 1
        if self.n_tokens < window:
            raise ValueError(
                f"shard has {self.n_tokens} tokens; need at least {window} "
                f"for context_length={context_length}"
            )
        self._max_start = self.n_tokens - window
        self._rng = np.random.default_rng(seed)
        # Epoch length: default to one pass worth of non-overlapping windows
        if num_samples is None:
            self._num_samples = max(1, self.n_tokens // window)
        else:
            if num_samples < 1:
                raise ValueError("num_samples must be >= 1")
            self._num_samples = num_samples

        log.info(
            "dataset_opened",
            path=str(self.shard_path),
            tokens=self.n_tokens,
            context=context_length,
            num_samples=self._num_samples,
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # Deterministic-but-varied starts: mix index with RNG stream
        start = int(self._rng.integers(0, self._max_start + 1))
        # Also fold index so different indices in one epoch differ
        start = (start + index * 9973) % (self._max_start + 1)
        window = self.context_length + 1
        chunk = np.array(self.data[start : start + window], dtype=np.int64)
        input_ids = torch.from_numpy(chunk[:-1].copy())
        labels = torch.from_numpy(chunk[1:].copy())
        return {"input_ids": input_ids, "labels": labels}


class MultiShardDataset(Dataset[dict[str, torch.Tensor]]):
    """Sample windows from multiple memmap shards (uniform over shards, then offset)."""

    def __init__(
        self,
        shard_paths: Sequence[Path | str],
        *,
        context_length: int = 1024,
        seed: int = 42,
        num_samples: int | None = None,
    ) -> None:
        if not shard_paths:
            raise ValueError("shard_paths must be non-empty")
        self.context_length = context_length
        self.shards: list[TokenShardDataset] = []
        for i, p in enumerate(shard_paths):
            self.shards.append(
                TokenShardDataset(
                    p,
                    context_length=context_length,
                    seed=seed + i * 1009,
                    num_samples=None,
                )
            )
        total_tokens = sum(s.n_tokens for s in self.shards)
        window = context_length + 1
        if num_samples is None:
            self._num_samples = max(1, total_tokens // window)
        else:
            if num_samples < 1:
                raise ValueError("num_samples must be >= 1")
            self._num_samples = num_samples
        self._rng = np.random.default_rng(seed)
        log.info(
            "multi_shard_opened",
            n_shards=len(self.shards),
            total_tokens=total_tokens,
            num_samples=self._num_samples,
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        shard_idx = int(self._rng.integers(0, len(self.shards)))
        # re-seed per index for variety while staying deterministic given seed stream
        return self.shards[shard_idx][index]


def worker_init_fn(worker_id: int) -> None:
    """Seed numpy / torch per DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)
    torch.manual_seed(worker_seed + worker_id)


def build_dataloader(
    shard_paths: Sequence[Path | str],
    *,
    context_length: int,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a multi-worker DataLoader over memmap shards."""
    ds: Dataset[dict[str, torch.Tensor]]
    if len(shard_paths) == 1:
        ds = TokenShardDataset(
            shard_paths[0],
            context_length=context_length,
            seed=seed,
        )
    else:
        ds = MultiShardDataset(
            shard_paths,
            context_length=context_length,
            seed=seed,
        )
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,  # random sampling is inside dataset
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        generator=gen,
        drop_last=True,
    )


def infinite_loader(
    loader: DataLoader[dict[str, torch.Tensor]],
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches forever (for step-based training)."""
    while True:
        yield from loader


def build_batch(
    shard_path: Path | str,
    *,
    context: int = 1024,
    batch: int = 4,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Sample a single batch of shape ``(batch, context)`` for smoke tests."""
    ds = TokenShardDataset(shard_path, context_length=context, seed=seed, num_samples=batch)
    inputs: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for i in range(batch):
        item = ds[i]
        inputs.append(item["input_ids"])
        labels.append(item["labels"])
    input_ids = torch.stack(inputs, dim=0)
    label_ids = torch.stack(labels, dim=0)
    return {"input_ids": input_ids, "labels": label_ids}


def _assert_batch(batch_out: dict[str, torch.Tensor], batch: int, context: int) -> None:
    x = batch_out["input_ids"]
    y = batch_out["labels"]
    if tuple(x.shape) != (batch, context):
        raise AssertionError(f"input_ids shape {tuple(x.shape)} != {(batch, context)}")
    if tuple(y.shape) != (batch, context):
        raise AssertionError(f"labels shape {tuple(y.shape)} != {(batch, context)}")
    # labels are input_ids shifted by one within each window:
    # for a window w[0..T], input=w[0:T], labels=w[1:T+1]
    # so labels[:, :-1] == input_ids[:, 1:]
    if not torch.equal(y[:, :-1], x[:, 1:]):
        raise AssertionError("labels are not input_ids shifted by one")
    if torch.isnan(x.float()).any() or torch.isnan(y.float()).any():
        raise AssertionError("NaN found in batch tensors")


def main(argv: list[str] | None = None) -> int:
    """CLI smoke test: ``python -m training.dataset --shard ... --context 1024 --batch 4``."""
    parser = argparse.ArgumentParser(description="Memmap token-shard smoke test")
    parser.add_argument("--shard", type=Path, required=True, help="Path to int32 .bin shard")
    parser.add_argument("--context", type=int, default=1024, help="Context length T")
    parser.add_argument("--batch", type=int, default=4, help="Batch size B")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    args = parser.parse_args(argv)

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )

    try:
        batch_out = build_batch(
            args.shard,
            context=args.context,
            batch=args.batch,
            seed=args.seed,
        )
        _assert_batch(batch_out, args.batch, args.context)
    except Exception:
        log.exception("dataset_smoke_failed")
        return 1

    log.info(
        "dataset_smoke_ok",
        shard=str(args.shard),
        input_shape=list(batch_out["input_ids"].shape),
        labels_shape=list(batch_out["labels"].shape),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
