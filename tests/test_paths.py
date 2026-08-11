"""Tests for DataPaths layout."""

from __future__ import annotations

from pathlib import Path

from minigpt_llm.data.paths import DataPaths


def test_paths_layout(tmp_path: Path) -> None:
    p = DataPaths(tmp_path)
    p.ensure_dirs()
    assert p.raw.is_dir()
    assert p.cleaned.is_dir()
    assert p.tokenized.is_dir()
    assert p.vocab.is_dir()
    assert p.tinystories_raw.name == "tinystories.txt"
    assert p.meta_pkl.name == "meta.pkl"
