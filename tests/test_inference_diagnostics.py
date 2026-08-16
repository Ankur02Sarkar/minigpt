"""Phase 5.3 diagnostics tests.

Skips cleanly when torch is unavailable (thin-client Mac, Python 3.13).

Uses the session-scoped ``tiny_setup`` fixture from ``tests/conftest.py``
(tiny BPE vocab 128 + 2-layer 64-dim GPT).
"""

from __future__ import annotations

import math

import pytest

# --torch-availability guard--------------------------------------------------
# Use pytest.importorskip at module level so the entire module is skipped
# when torch is not available.  This matches the project convention.
# -----------------------------------------------------------------------------
pytest.importorskip("torch")  # type: ignore[no-redef]

from inference.diagnostics import compute_entropy, DiagnosticAccumulator, on_token_default  # noqa: F401


@pytest.fixture(scope="session")
def tiny_setup():
    """Session-scoped tiny model + tokenizer fixture.

    The fixture body imports torch via ``pytest.importorskip`` pattern so that
    module-level collection can proceed on thin-client Mac; tests that do not
    request this fixture keep running.
    """
    from tokenizers import ByteLevelBPETokenizer  # noqa: E402

    from minigpt_llm.model.config import ModelConfig  # noqa: E402
    from minigpt_llm.model.model import GPT  # noqa: E402
    from minigpt_llm.tokenizer.load import load_tokenizer  # noqa: E402

    tmp = __import__("tempfile").TemporaryDirectory()
    d = tmp.name
    corpus = [
        "the quick brown fox jumps over the lazy dog",
        "once upon a time there was a small language model",
        "two roads diverged in a yellow wood and i",
        "the model the model the model again and again",
        "stop here and then keep going after the end",
    ]
    with open(f"{d}/corpus.txt", "w") as f:
        f.write("\n".join(corpus))

    tok = ByteLevelBPETokenizer()
    tok.train([f"{d}/corpus.txt"], vocab_size=128, min_frequency=1,
              special_tokens=["<pad>", "", "asures", "▁"])
    tok.save_model(d)
    tokenizer = load_tokenizer(d)

    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=tokenizer.get_vocab_size(),
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        max_position_embeddings=64,
        dropout=0.0,
        tie_weights=True,
    )
    model = GPT(cfg)
    model.eval()
    return torch, tokenizer, tmp