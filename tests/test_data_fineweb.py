"""Tests for FineWeb stream filter + token cap (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from minigpt_llm.data.fineweb import (
    filter_doc,
    iter_fineweb_docs,
    stream_and_tokenize_fineweb,
)
from minigpt_llm.data.paths import DataPaths


def test_filter_doc_length() -> None:
    assert filter_doc("x" * 100, require_english=False) is False
    assert filter_doc("x" * 250, require_english=False) is True
    assert filter_doc("x" * 100_001, require_english=False) is False


def test_filter_doc_english(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_english(text: str, *, min_score: float = 0.9) -> bool:
        return "english" in text.lower()

    monkeypatch.setattr("minigpt_llm.data.fineweb.is_english", fake_english)
    assert filter_doc("x" * 250 + " english text", require_english=True) is True
    assert filter_doc("x" * 250 + " other", require_english=True) is False


def test_iter_fineweb_docs_single_buffer() -> None:
    rows = [{"text": "a" * 250, "id": "1"}, {"text": "b" * 250, "id": "2"}]

    def load_fn(*_a, **_k):  # type: ignore[no-untyped-def]
        return rows

    out = list(iter_fineweb_docs(load_fn=load_fn, max_docs_buffered=1))
    assert out == [("1", "a" * 250), ("2", "b" * 250)]


def test_stream_and_tokenize_cap(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()

    docs = [(str(i), "x" * 250) for i in range(50)]

    def encode_fn(batch: list[str]) -> list[list[int]]:
        # 10 tokens per doc
        return [[i % 100 for i in range(10)] for _ in batch]

    stats = stream_and_tokenize_fineweb(
        paths,
        encode_fn,
        token_cap=35,
        batch_docs=5,
        force=True,
        require_english=False,
        doc_iter=docs,
        eos_id=None,
    )
    assert stats["tokens"] == 35
    assert stats["hit_cap"] is True
    assert paths.fineweb_bin.stat().st_size == 35 * 4
    meta = paths.meta_pkl
    assert meta.is_file()


def test_stream_refuses_overwrite(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()
    paths.fineweb_bin.write_bytes(b"\x00\x00\x00\x00")

    def encode_fn(batch: list[str]) -> list[list[int]]:
        return [[1, 2, 3] for _ in batch]

    with pytest.raises(FileExistsError):
        stream_and_tokenize_fineweb(
            paths,
            encode_fn,
            token_cap=10,
            require_english=False,
            doc_iter=[("1", "x" * 250)],
            force=False,
        )
