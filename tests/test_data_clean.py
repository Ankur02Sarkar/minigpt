"""Tests for clean + dedupe pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from minigpt_llm.data.clean import clean_and_dedupe_files, clean_line, sha1_normalized
from minigpt_llm.data.paths import DataPaths


def test_clean_line_strips_controls_and_ws() -> None:
    assert clean_line("  hello\x00  world\t\n") == "hello world"
    assert clean_line("") == ""
    assert clean_line("   \n") == ""


def test_sha1_stable() -> None:
    a = sha1_normalized("hello world")
    b = sha1_normalized("hello world")
    assert a == b
    assert len(a) == 40


def test_clean_and_dedupe(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()
    raw_a = paths.raw / "a.txt"
    raw_b = paths.raw / "b.txt"
    raw_a.write_text("Hello World\nhello world\n\nfoo\n", encoding="utf-8")
    raw_b.write_text("foo\nbar\n", encoding="utf-8")

    stats = clean_and_dedupe_files(paths, inputs=[raw_a, raw_b], force=True)
    # clean_line does not lower-case: "Hello World" != "hello world"
    # From a: Hello World, hello world, foo; empty skipped
    # From b: foo (dupe), bar → kept 4, duplicates 1
    assert stats["kept"] == 4
    assert stats["duplicates_dropped"] == 1
    text = paths.all_text_cleaned.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln]
    assert lines == ["Hello World", "hello world", "foo", "bar"]


def test_clean_refuses_overwrite(tmp_path: Path) -> None:
    paths = DataPaths(tmp_path)
    paths.ensure_dirs()
    raw = paths.raw / "a.txt"
    raw.write_text("x\n", encoding="utf-8")
    clean_and_dedupe_files(paths, inputs=[raw], force=True)
    with pytest.raises(FileExistsError):
        clean_and_dedupe_files(paths, inputs=[raw], force=False)
