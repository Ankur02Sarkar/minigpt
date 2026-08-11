"""Load a trained ByteLevel BPE tokenizer from disk."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from tokenizers import ByteLevelBPETokenizer

from minigpt_llm.data.paths import DataPaths
from minigpt_llm.tokenizer.train_bpe import SPECIAL_TOKENS

__all__ = ["eos_token_id", "load_tokenizer", "make_encode_batch"]


def load_tokenizer(vocab_dir: Path | DataPaths) -> ByteLevelBPETokenizer:
    """Load ``vocab.json`` + ``merges.txt`` from a vocab directory or DataPaths."""
    directory = vocab_dir.vocab if isinstance(vocab_dir, DataPaths) else Path(vocab_dir)
    vocab = directory / "vocab.json"
    merges = directory / "merges.txt"
    if not vocab.is_file() or not merges.is_file():
        raise FileNotFoundError(
            f"tokenizer files missing under {directory} (need vocab.json + merges.txt)"
        )
    tok = ByteLevelBPETokenizer(str(vocab), str(merges))
    # Ensure special tokens are registered if present in vocab
    existing = set(tok.get_vocab().keys())
    for t in SPECIAL_TOKENS:
        if t not in existing:
            tok.add_special_tokens([t])
    return tok


def make_encode_batch(tok: ByteLevelBPETokenizer) -> Callable[[list[str]], list[list[int]]]:
    """Return ``encode_fn(list[str]) -> list[list[int]]`` for shard writers."""

    def encode_fn(batch: list[str]) -> list[list[int]]:
        return [enc.ids for enc in tok.encode_batch(batch)]

    return encode_fn


def eos_token_id(tok: ByteLevelBPETokenizer) -> int | None:
    """Return ``</s>`` id if present in vocab."""
    return tok.get_vocab().get("</s>")
