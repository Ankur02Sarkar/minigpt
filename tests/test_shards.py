"""Tests for int32 shard write/read and val split."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from minigpt_llm.data.paths import DataPaths
from minigpt_llm.data.shards import (
    append_token_ids,
    load_meta,
    save_meta,
    split_wikitext_val,
    tokenize_text_file,
    write_token_ids,
)


def test_write_and_append(tmp_path: Path) -> None:
    path = tmp_path / "a.bin"
    n = write_token_ids(path, [1, 2, 3], force=True)
    assert n == 3
    n2 = append_token_ids(path, [4, 5])
    assert n2 == 2
    data = np.fromfile(path, dtype=np.int32)
    assert data.tolist() == [1, 2, 3, 4, 5]


def test_meta_roundtrip(tmp_path: Path) -> None:
    meta_path = tmp_path / "meta.pkl"
    save_meta(meta_path, {"tinystories": {"tokens": 10, "dtype": "int32"}})
    loaded = load_meta(meta_path)
    assert loaded["tinystories"]["tokens"] == 10


def test_tokenize_text_file(tmp_path: Path) -> None:
    text = tmp_path / "docs.txt"
    text.write_text("hello\nworld\n", encoding="utf-8")
    out = tmp_path / "out.bin"

    def encode_fn(batch: list[str]) -> list[list[int]]:
        return [[ord(c) % 100 for c in s] for s in batch]

    n = tokenize_text_file(text, out, encode_fn, force=True, add_eos=True, eos_id=0)
    assert n > 0
    data = np.fromfile(out, dtype=np.int32)
    assert 0 in data.tolist()  # eos appended


def test_split_wikitext_val(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()
    ids = list(range(100))
    write_token_ids(paths.wikitext_bin, ids, force=True)
    stats = split_wikitext_val(paths, train_frac=0.95, force=True)
    assert stats["train_tokens"] == 95
    assert stats["val_tokens"] == 5
    train = np.fromfile(paths.wikitext_bin, dtype=np.int32)
    val = np.fromfile(paths.val_bin, dtype=np.int32)
    assert train.tolist() == list(range(95))
    assert val.tolist() == list(range(95, 100))
    meta = load_meta(paths.meta_pkl)
    assert meta["val"]["tokens"] == 5
