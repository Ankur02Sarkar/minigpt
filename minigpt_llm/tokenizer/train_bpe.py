"""Train a ByteLevel BPE tokenizer (vocab 32k) on cleaned text."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog
from tokenizers import ByteLevelBPETokenizer

from minigpt_llm.data.paths import DataPaths

__all__ = ["SPECIAL_TOKENS", "train_bpe", "verify_roundtrip"]

log = structlog.get_logger(__name__)

SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>"]


def train_bpe(
    paths: DataPaths,
    *,
    files: list[Path] | None = None,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    force: bool = False,
) -> dict[str, Any]:
    """1.6 — Train BPE and save ``vocab.json`` + ``merges.txt`` under ``/data/vocab/``."""
    paths.ensure_dirs()
    if files is None:
        files = [paths.all_text_cleaned]
    for f in files:
        if not f.is_file():
            raise FileNotFoundError(f"missing training file for BPE: {f}")

    vocab_json = paths.vocab / "vocab.json"
    merges_txt = paths.vocab / "merges.txt"
    if (vocab_json.exists() or merges_txt.exists()) and not force:
        raise FileExistsError(
            f"tokenizer files already exist in {paths.vocab}; pass force=True to overwrite"
        )

    log.info(
        "bpe_train_start",
        files=[str(f) for f in files],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
    )
    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[str(f) for f in files],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    paths.vocab.mkdir(parents=True, exist_ok=True)
    tok.save_model(str(paths.vocab))

    # HF save_model writes vocab.json + merges.txt (no prefix).
    if not vocab_json.is_file() or not merges_txt.is_file():
        raise RuntimeError(f"tokenizer save failed; expected files in {paths.vocab}")

    sample_stats = verify_roundtrip(tok, files[0], n_samples=min(1000, _count_lines(files[0])))
    stats: dict[str, Any] = {
        "vocab_dir": str(paths.vocab),
        "vocab_size": tok.get_vocab_size(),
        "vocab_json": str(vocab_json),
        "merges_txt": str(merges_txt),
        **sample_stats,
    }
    log.info("bpe_train_done", **stats)
    return stats


def verify_roundtrip(
    tok: ByteLevelBPETokenizer,
    text_path: Path,
    *,
    n_samples: int = 1000,
) -> dict[str, Any]:
    """Encode/decode a sample of lines; count exact-match round-trips.

    Byte-level BPE is lossy around whitespace normalization in some edge cases;
    we report match rate rather than hard-failing on partial mismatches.
    """
    checked = 0
    exact = 0
    with text_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            ids = tok.encode(line).ids
            decoded = tok.decode(ids)
            checked += 1
            if decoded == line or decoded.strip() == line.strip():
                exact += 1
            if checked >= n_samples:
                break
    if checked == 0:
        raise RuntimeError(f"no sample lines in {text_path} for round-trip check")
    rate = exact / checked
    log.info("bpe_roundtrip", checked=checked, exact=exact, rate=rate)
    if rate < 0.95:
        raise RuntimeError(f"BPE round-trip match rate too low: {rate:.3f} ({exact}/{checked})")
    return {"roundtrip_checked": checked, "roundtrip_exact": exact, "roundtrip_rate": rate}


def _count_lines(path: Path, limit: int = 10_000) -> int:
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in f:
            n += 1
            if n >= limit:
                break
    return n
