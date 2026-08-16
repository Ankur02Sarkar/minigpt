"""Shared pytest fixtures for the inference test suite.

Provides a session-scoped ``tiny_setup`` fixture: a tiny ByteLevel BPE
tokenizer (vocab 128) trained on a small corpus + a tiny 2-layer 64-dim GPT,
mirroring the ``tests/test_overfit.py`` CPU pattern. Tests under ``tests/``
that need a model + tokenizer import this fixture (pytest auto-discovers
``conftest.py``) instead of re-building one each time.
"""

from __future__ import annotations

import os
import tempfile

import pytest

# NOTE: torch + model imports are deferred into the fixture body so that this
# conftest module loads cleanly on torch-free thin clients (Mac Python 3.13).
# Tests that don't request ``tiny_setup`` keep running; only tests that need
# the model pay the (skipped) torch import cost.

# Same corpus as tests/test_inference_generate.py originally used; kept here so
# all inference tests share an identical tiny model for reproducibility.
_TINY_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "once upon a time there was a small language model",
    "two roads diverged in a yellow wood and i",
    "the model the model the model again and again",
    "stop here and then keep going after the end",
]


@pytest.fixture(scope="session")
def tiny_setup():
    """Train a tiny BPE tokenizer + instantiate a tiny GPT sized to match it.

    Returns ``(model, tokenizer, tmpdir)`` — the tmpdir is kept alive for the
    session so the tokenizer files on disk remain readable. Tests that don't
    need the tmpdir can ignore the third element.

    Skips cleanly when torch is unavailable (thin-client Mac) — the import is
    deferred to the fixture body so non-torch tests in the suite keep running.
    """
    torch = pytest.importorskip("torch")
    from tokenizers import ByteLevelBPETokenizer

    from minigpt_llm.model.config import ModelConfig
    from minigpt_llm.model.model import GPT
    from minigpt_llm.tokenizer.load import load_tokenizer

    tmp = tempfile.TemporaryDirectory()
    d = tmp.name
    src = os.path.join(d, "corpus.txt")
    with open(src, "w") as f:
        f.write("\n".join(_TINY_CORPUS))
    tok = ByteLevelBPETokenizer()
    tok.train(
        [src],
        vocab_size=128,
        min_frequency=1,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>"],
    )
    tok.save_model(d)
    tokenizer = load_tokenizer(d)
    vocab_size = tokenizer.get_vocab_size()
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=vocab_size,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        max_position_embeddings=64,
        dropout=0.0,
        tie_weights=True,
    )
    model = GPT(cfg)
    model.eval()
    return model, tokenizer, tmp
