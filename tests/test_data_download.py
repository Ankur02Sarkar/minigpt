"""Tests for raw download writers (mocked datasets)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from minigpt_llm.data.download import (
    download_tinystories,
    download_wikitext,
    normalize_doc,
    write_docs_to_txt,
)
from minigpt_llm.data.paths import DataPaths


def test_normalize_doc() -> None:
    assert normalize_doc("a\n\nb  c") == "a b c"


def test_write_docs_to_txt(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    meta = write_docs_to_txt(iter(["hello", "", "world\nline"]), out)
    assert meta["lines"] == 2
    assert out.read_text(encoding="utf-8") == "hello\nworld line\n"
    assert len(meta["sha256"]) == 64


def test_write_docs_refuses_overwrite(tmp_path: Path) -> None:
    out = tmp_path / "out.txt"
    write_docs_to_txt(iter(["a"]), out)
    with pytest.raises(FileExistsError):
        write_docs_to_txt(iter(["b"]), out, force=False)


def _fake_load_dataset(*_args: Any, **_kwargs: Any) -> list[dict[str, str]]:
    return [{"text": "Story one."}, {"text": "Story two."}, {"text": ""}]


def test_download_tinystories_mocked(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    meta = download_tinystories(paths, force=True, load_fn=_fake_load_dataset)
    assert meta["lines"] == 2
    assert paths.tinystories_raw.is_file()


def test_download_wikitext_mocked(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    meta = download_wikitext(paths, force=True, load_fn=_fake_load_dataset)
    assert meta["lines"] == 2
    assert paths.wikitext_raw.is_file()
