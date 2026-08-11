"""Download TinyStories and WikiText-103 to plain-text files on the data disk."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import structlog

from minigpt_llm.data.paths import DataPaths

__all__ = [
    "download_tinystories",
    "download_wikitext",
    "file_sha256",
    "normalize_doc",
    "write_docs_to_txt",
]

log = structlog.get_logger(__name__)


def normalize_doc(text: str) -> str:
    """Collapse a multi-line document into a single line for one-doc-per-line storage."""
    return " ".join(text.split())


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file in streaming chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def write_docs_to_txt(
    docs: Iterator[str],
    out_path: Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Write one normalized document per line. Refuses to overwrite unless force."""
    if out_path.exists() and not force:
        raise FileExistsError(f"{out_path} already exists; pass force=True to overwrite")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    n_lines = 0
    n_empty_skipped = 0
    with tmp.open("w", encoding="utf-8") as f:
        for raw in docs:
            line = normalize_doc(raw)
            if not line:
                n_empty_skipped += 1
                continue
            f.write(line)
            f.write("\n")
            n_lines += 1
        f.flush()
    tmp.replace(out_path)
    size_bytes = out_path.stat().st_size
    digest = file_sha256(out_path)
    meta = {
        "path": str(out_path),
        "lines": n_lines,
        "empty_skipped": n_empty_skipped,
        "bytes": size_bytes,
        "sha256": digest,
    }
    log.info("wrote_raw_txt", **meta)
    return meta


def _iter_dataset_text(
    dataset_id: str,
    *,
    config: str | None = None,
    split: str = "train",
    text_field: str = "text",
    load_fn: Callable[..., Any] | None = None,
) -> Iterator[str]:
    if load_fn is None:
        from datasets import load_dataset

        load_fn = load_dataset
    if config is None:
        ds = load_fn(dataset_id, split=split)
    else:
        ds = load_fn(dataset_id, config, split=split)
    for row in ds:
        text = row.get(text_field)
        if text is None:
            continue
        if not isinstance(text, str):
            text = str(text)
        yield text


def download_tinystories(
    paths: DataPaths,
    *,
    force: bool = False,
    load_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """1.1 — Download TinyStories train split → ``/data/raw/tinystories.txt``."""
    paths.ensure_dirs()
    log.info("download_tinystories_start", out=str(paths.tinystories_raw))
    docs = _iter_dataset_text(
        "roneneldan/TinyStories",
        split="train",
        text_field="text",
        load_fn=load_fn,
    )
    return write_docs_to_txt(docs, paths.tinystories_raw, force=force)


def download_wikitext(
    paths: DataPaths,
    *,
    force: bool = False,
    load_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """1.2 — Download WikiText-103 train split → ``/data/raw/wikitext103.txt``."""
    paths.ensure_dirs()
    log.info("download_wikitext_start", out=str(paths.wikitext_raw))
    # Prefer namespaced id: huggingface_hub>=1.x rejects bare "wikitext" (no namespace).
    docs = _iter_dataset_text(
        "Salesforce/wikitext",
        config="wikitext-103-v1",
        split="train",
        text_field="text",
        load_fn=load_fn,
    )
    return write_docs_to_txt(docs, paths.wikitext_raw, force=force)
