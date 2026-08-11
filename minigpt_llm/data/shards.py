"""Write/read int32 token shards and meta.pkl."""

from __future__ import annotations

import pickle
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from minigpt_llm.data.paths import DataPaths

__all__ = [
    "ShardMeta",
    "append_token_ids",
    "load_meta",
    "save_meta",
    "split_wikitext_val",
    "tokenize_text_file",
    "write_token_ids",
]

log = structlog.get_logger(__name__)

ShardMeta = dict[str, dict[str, Any]]

_DTYPE = np.dtype("<i4")  # little-endian int32


def load_meta(path: Path) -> ShardMeta:
    """Load ``meta.pkl``; empty dict if missing."""
    if not path.is_file():
        return {}
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"meta.pkl must be a dict, got {type(data)}")
    return data


def save_meta(path: Path, meta: ShardMeta) -> None:
    """Atomic pickle write of shard metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
    tmp.replace(path)
    log.info("meta_saved", path=str(path), shards=list(meta.keys()))


def write_token_ids(
    path: Path,
    token_ids: Sequence[int] | np.ndarray,
    *,
    force: bool = False,
) -> int:
    """Write a full int32 shard atomically. Returns token count."""
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass force=True to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(token_ids, dtype=_DTYPE)
    if arr.ndim != 1:
        raise ValueError(f"token_ids must be 1-D, got shape {arr.shape}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    arr.tofile(tmp)
    tmp.replace(path)
    n = int(arr.shape[0])
    log.info("shard_written", path=str(path), tokens=n, bytes=n * 4)
    return n


def append_token_ids(path: Path, token_ids: Sequence[int] | np.ndarray) -> int:
    """Append int32 IDs to an existing shard (or create it). Returns count appended."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(token_ids, dtype=_DTYPE)
    if arr.ndim != 1:
        raise ValueError(f"token_ids must be 1-D, got shape {arr.shape}")
    with path.open("ab") as f:
        arr.tofile(f)
        f.flush()
    n = int(arr.shape[0])
    log.debug("shard_appended", path=str(path), tokens=n)
    return n


def tokenize_text_file(
    text_path: Path,
    out_bin: Path,
    encode_fn: Any,
    *,
    batch_lines: int = 1000,
    force: bool = False,
    add_eos: bool = True,
    eos_id: int | None = None,
) -> int:
    """Encode a one-doc-per-line text file into a contiguous int32 shard.

    ``encode_fn(list[str]) -> list[list[int]]`` should match HF tokenizers
    ``encode_batch`` style (list of id lists).
    """
    if not text_path.is_file():
        raise FileNotFoundError(f"missing text file: {text_path}")
    if out_bin.exists() and not force:
        raise FileExistsError(f"{out_bin} already exists; pass force=True to overwrite")
    if out_bin.exists() and force:
        out_bin.unlink()

    total = 0
    batch: list[str] = []

    def flush() -> None:
        nonlocal total, batch
        if not batch:
            return
        encoded = encode_fn(batch)
        ids: list[int] = []
        for seq in encoded:
            ids.extend(seq)
            if add_eos and eos_id is not None:
                ids.append(eos_id)
        total += append_token_ids(out_bin, ids)
        batch = []

    with text_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            batch.append(line)
            if len(batch) >= batch_lines:
                flush()
        flush()

    if total == 0:
        raise RuntimeError(f"no tokens written from {text_path}")
    log.info("tokenize_text_done", src=str(text_path), out=str(out_bin), tokens=total)
    return total


def split_wikitext_val(
    paths: DataPaths,
    *,
    train_frac: float = 0.95,
    force: bool = False,
) -> dict[str, int]:
    """1.8 — Split tokenized WikiText 95/5 into ``wikitext.bin`` + ``val.bin``.

    Reads the existing ``wikitext.bin`` (full), rewrites train prefix and val suffix.
    """
    src = paths.wikitext_bin
    if not src.is_file():
        raise FileNotFoundError(f"wikitext shard missing: {src}")
    if train_frac <= 0.0 or train_frac >= 1.0:
        raise ValueError(f"train_frac must be in (0,1), got {train_frac}")

    data = np.memmap(src, dtype=_DTYPE, mode="r")
    n = int(data.shape[0])
    if n < 2:
        raise RuntimeError(f"wikitext shard too small to split: {n} tokens")
    n_train = max(1, int(n * train_frac))
    n_val = n - n_train
    if n_val < 1:
        n_train = n - 1
        n_val = 1

    train_ids = np.array(data[:n_train], dtype=_DTYPE)
    val_ids = np.array(data[n_train:], dtype=_DTYPE)
    # Release memmap before rewrite
    del data

    write_token_ids(paths.wikitext_bin, train_ids, force=True)
    write_token_ids(paths.val_bin, val_ids, force=force or True)

    meta = load_meta(paths.meta_pkl)
    meta["wikitext"] = {"tokens": n_train, "dtype": "int32"}
    meta["val"] = {"tokens": n_val, "dtype": "int32"}
    save_meta(paths.meta_pkl, meta)

    stats = {"train_tokens": n_train, "val_tokens": n_val, "total_before": n}
    log.info("wikitext_val_split", **stats)
    return stats


def update_meta_for_shards(
    paths: DataPaths,
    shard_paths: Iterable[tuple[str, Path]],
) -> ShardMeta:
    """Recompute meta entries from on-disk file sizes (tokens = bytes // 4)."""
    meta = load_meta(paths.meta_pkl)
    for name, path in shard_paths:
        if not path.is_file():
            continue
        n = path.stat().st_size // 4
        meta[name] = {"tokens": n, "dtype": "int32"}
    save_meta(paths.meta_pkl, meta)
    return meta
