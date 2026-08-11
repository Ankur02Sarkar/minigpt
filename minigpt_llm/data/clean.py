"""Clean and document-level dedupe for TinyStories + WikiText."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import regex as re
import structlog

from minigpt_llm.data.paths import DataPaths

__all__ = [
    "clean_line",
    "clean_and_dedupe_files",
    "iter_cleaned_docs",
    "sha1_normalized",
]

log = structlog.get_logger(__name__)

# Strip C0/C1 controls (tabs/newlines are collapsed by whitespace pass).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]+")
_WS = re.compile(r"\s+")


def clean_line(text: str) -> str:
    """Strip control chars, collapse whitespace, strip ends. Empty → ''."""
    if not text:
        return ""
    text = _CONTROL.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def sha1_normalized(text: str) -> str:
    """SHA-1 of cleaned/normalized text for doc-level dedupe."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_cleaned_docs(paths: Iterable[Path]) -> Iterator[str]:
    """Yield cleaned non-empty lines from one-doc-per-line text files."""
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing input for clean pass: {path}")
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                cleaned = clean_line(line)
                if cleaned:
                    yield cleaned


def clean_and_dedupe_files(
    paths: DataPaths,
    *,
    inputs: list[Path] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """1.5 — Clean + SHA-1 dedupe TinyStories + WikiText → ``all_text.txt``.

    Returns stats including raw/kept counts and dedupe ratio.
    """
    paths.ensure_dirs()
    out = paths.all_text_cleaned
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists; pass force=True to overwrite")

    if inputs is None:
        inputs = [paths.tinystories_raw, paths.wikitext_raw]

    seen: set[str] = set()
    n_in = 0
    n_empty = 0
    n_dupes = 0
    n_kept = 0
    tmp = out.with_suffix(out.suffix + ".tmp")

    with tmp.open("w", encoding="utf-8") as fout:
        for path in inputs:
            if not path.is_file():
                raise FileNotFoundError(f"missing raw file: {path}")
            log.info("clean_file_start", path=str(path))
            with path.open("r", encoding="utf-8", errors="replace") as fin:
                for line in fin:
                    n_in += 1
                    cleaned = clean_line(line)
                    if not cleaned:
                        n_empty += 1
                        continue
                    digest = sha1_normalized(cleaned)
                    if digest in seen:
                        n_dupes += 1
                        continue
                    seen.add(digest)
                    fout.write(cleaned)
                    fout.write("\n")
                    n_kept += 1
            log.info("clean_file_done", path=str(path), kept_so_far=n_kept)
        fout.flush()

    tmp.replace(out)
    size_bytes = out.stat().st_size
    non_empty = n_in - n_empty
    dedupe_ratio = (n_dupes / non_empty) if non_empty else 0.0
    stats: dict[str, Any] = {
        "path": str(out),
        "input_lines": n_in,
        "empty_dropped": n_empty,
        "duplicates_dropped": n_dupes,
        "kept": n_kept,
        "dedupe_ratio": dedupe_ratio,
        "bytes": size_bytes,
    }
    log.info("clean_dedupe_done", **stats)
    return stats
