"""Validation loss / perplexity over a memmap shard."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn as nn

__all__ = ["evaluate_shard"]

log = structlog.get_logger(__name__)

_DTYPE = np.dtype("<i4")


@torch.no_grad()
def evaluate_shard(
    model: nn.Module,
    shard_path: Path | str,
    *,
    context_length: int,
    batch_size: int = 4,
    max_batches: int | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Average CE loss over sequential non-overlapping windows; return loss + ppl."""
    path = Path(shard_path)
    if not path.is_file():
        raise FileNotFoundError(f"val shard not found: {path}")

    data = np.memmap(path, dtype=_DTYPE, mode="r")
    n = int(data.shape[0])
    window = context_length + 1
    if n < window:
        raise ValueError(f"val shard too short: {n} tokens < {window}")

    model.eval()
    device = torch.device(device)
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    starts = list(range(0, n - window + 1, context_length))

    batch_x: list[torch.Tensor] = []
    batch_y: list[torch.Tensor] = []

    def flush() -> None:
        nonlocal total_loss, total_tokens, n_batches, batch_x, batch_y
        if not batch_x:
            return
        input_ids = torch.stack(batch_x, dim=0).to(device)
        labels = torch.stack(batch_y, dim=0).to(device)
        out = model(input_ids, labels=labels)
        if out.loss is None:
            raise RuntimeError("model returned no loss during eval")
        # loss is mean over tokens in batch
        bsz = input_ids.size(0)
        tok = bsz * context_length
        total_loss += float(out.loss.item()) * tok
        total_tokens += tok
        n_batches += 1
        batch_x = []
        batch_y = []

    for start in starts:
        if max_batches is not None and n_batches >= max_batches and not batch_x:
            break
        chunk = np.array(data[start : start + window], dtype=np.int64)
        batch_x.append(torch.from_numpy(chunk[:-1].copy()))
        batch_y.append(torch.from_numpy(chunk[1:].copy()))
        if len(batch_x) >= batch_size:
            flush()
            if max_batches is not None and n_batches >= max_batches:
                break
    flush()

    if total_tokens == 0:
        raise RuntimeError("evaluation produced zero tokens")
    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 100.0))  # clamp for overflow
    metrics = {"val_loss": avg_loss, "val_ppl": ppl, "val_tokens": float(total_tokens)}
    log.info("eval_done", path=str(path), **metrics)
    return metrics
