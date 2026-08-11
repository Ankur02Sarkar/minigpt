"""FineWeb-Edu streaming scaffold with filter + capped on-the-fly tokenization."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

import structlog

from minigpt_llm.data.paths import DataPaths
from minigpt_llm.data.shards import append_token_ids, load_meta, save_meta

__all__ = [
    "FINEWEB_TOKEN_CAP",
    "filter_doc",
    "is_english",
    "iter_fineweb_docs",
    "stream_and_tokenize_fineweb",
]

log = structlog.get_logger(__name__)

FINEWEB_TOKEN_CAP = 500_000_000
MIN_CHARS = 200
MAX_CHARS = 100_000
LANG_MIN_SCORE = 0.9
DEFAULT_BATCH_DOCS = 1000


def is_english(text: str, *, min_score: float = LANG_MIN_SCORE) -> bool:
    """Return True if langdetect assigns English probability >= min_score."""
    try:
        from langdetect import LangDetectException, detect_langs
    except ImportError as exc:  # pragma: no cover
        raise ImportError("langdetect is required for FineWeb language filtering") from exc

    try:
        langs = detect_langs(text[:5000])  # cap work on long docs
    except LangDetectException:
        return False
    return any(lang.lang == "en" and float(lang.prob) >= min_score for lang in langs)


def filter_doc(
    text: str,
    *,
    min_chars: int = MIN_CHARS,
    max_chars: int = MAX_CHARS,
    require_english: bool = True,
    lang_min_score: float = LANG_MIN_SCORE,
) -> bool:
    """Return True if the document should be kept for tokenization."""
    if not text or not isinstance(text, str):
        return False
    n = len(text)
    if n < min_chars or n > max_chars:
        return False
    if not require_english:
        return True
    return is_english(text, min_score=lang_min_score)


def iter_fineweb_docs(
    *,
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    config: str = "sample-10BT",
    split: str = "train",
    streaming: bool = True,
    load_fn: Callable[..., Any] | None = None,
    max_docs_buffered: int = 1,
) -> Iterator[tuple[str, str]]:
    """1.3 — Yield ``(doc_id, text)`` without persisting raw FineWeb docs.

    ``max_docs_buffered`` is the intended in-flight buffer for tests; the
    generator itself only holds one row at a time.
    """
    if max_docs_buffered < 1:
        raise ValueError("max_docs_buffered must be >= 1")
    if load_fn is None:
        from datasets import load_dataset

        load_fn = load_dataset

    log.info(
        "fineweb_stream_open",
        dataset=dataset_name,
        config=config,
        split=split,
        streaming=streaming,
    )
    ds = load_fn(dataset_name, name=config, split=split, streaming=streaming)
    for i, row in enumerate(ds):
        text = row.get("text", "")
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        doc_id = str(row.get("id", row.get("dump", i)))
        yield doc_id, text


def stream_and_tokenize_fineweb(
    paths: DataPaths,
    encode_fn: Callable[[list[str]], list[list[int]]],
    *,
    token_cap: int = FINEWEB_TOKEN_CAP,
    batch_docs: int = DEFAULT_BATCH_DOCS,
    force: bool = False,
    require_english: bool = True,
    doc_iter: Iterable[tuple[str, str]] | None = None,
    eos_id: int | None = None,
) -> dict[str, Any]:
    """1.4 — Stream FineWeb, filter, tokenize, stop at ``token_cap``.

    Appends int32 IDs to ``paths.fineweb_bin``. Does **not** store raw docs.
    """
    paths.ensure_dirs()
    out = paths.fineweb_bin
    if out.exists() and not force:
        raise FileExistsError(f"{out} already exists; pass force=True to overwrite")
    if out.exists() and force:
        out.unlink()

    if doc_iter is None:
        doc_iter = iter_fineweb_docs()

    n_seen = 0
    n_kept = 0
    n_dropped = 0
    n_tokens = 0
    batch: list[str] = []
    # Guard: never hold more than batch_docs texts in memory
    max_buffered = batch_docs

    def flush() -> None:
        nonlocal batch, n_tokens, n_kept
        if not batch:
            return
        if len(batch) > max_buffered:
            raise RuntimeError(f"FineWeb buffer exceeded max_buffered={max_buffered}: {len(batch)}")
        encoded = encode_fn(batch)
        ids: list[int] = []
        for seq in encoded:
            ids.extend(seq)
            if eos_id is not None:
                ids.append(eos_id)
        # Trim if this batch would overshoot the hard cap
        remaining = token_cap - n_tokens
        if remaining <= 0:
            batch = []
            return
        if len(ids) > remaining:
            ids = ids[:remaining]
        n_tokens += append_token_ids(out, ids)
        n_kept += len(batch)
        batch = []

    for _doc_id, text in doc_iter:
        n_seen += 1
        if not filter_doc(text, require_english=require_english):
            n_dropped += 1
            continue
        batch.append(text)
        if len(batch) >= batch_docs:
            flush()
            if n_tokens >= token_cap:
                break
        if n_seen % 10_000 == 0:
            log.info(
                "fineweb_progress",
                seen=n_seen,
                kept=n_kept,
                dropped=n_dropped,
                tokens=n_tokens,
                cap=token_cap,
            )

    if n_tokens < token_cap and batch:
        flush()

    if n_tokens == 0:
        raise RuntimeError("FineWeb stream produced zero tokens; check filters / network")

    meta = load_meta(paths.meta_pkl)
    meta["fineweb"] = {"tokens": n_tokens, "dtype": "int32"}
    save_meta(paths.meta_pkl, meta)

    stats: dict[str, Any] = {
        "path": str(out),
        "docs_seen": n_seen,
        "docs_kept": n_kept,
        "docs_dropped": n_dropped,
        "tokens": n_tokens,
        "token_cap": token_cap,
        "hit_cap": n_tokens >= token_cap,
    }
    log.info("fineweb_tokenize_done", **stats)
    return stats
