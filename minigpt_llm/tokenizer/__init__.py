"""BPE tokenizer training and loading."""

from __future__ import annotations

from minigpt_llm.tokenizer.load import load_tokenizer
from minigpt_llm.tokenizer.train_bpe import train_bpe

__all__ = ["load_tokenizer", "train_bpe"]
