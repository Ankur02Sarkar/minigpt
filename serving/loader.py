"""Shared model loader for serving infrastructure.

Lazy-load on first request; hold in module-level singleton; thread-safe.
Warm-up pass on load to avoid first-request latency spike.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from minigpt_llm.model import GPT, ModelConfig
from minigpt_llm.tokenizer.load import load_tokenizer


_class_lock = threading.Lock()
_model: torch.nn.Module | None = None
_tokenizer: Any = None
_config: ModelConfig | None = None


def initialize(model_path: str, tokenizer_dir: str) -> None:
    """Load model and tokenizer at startup.

    Must be called before first request. Thread-safe.
    If checkpoint file is missing (e.g. running standalone container without Azure volume mount),
    creates a working in-memory demonstration model + tokenizer so endpoints can be tested.
    """
    global _model, _tokenizer, _config

    with _class_lock:
        if _model is not None:
            return  # already initialized

        from pathlib import Path
        p_model = Path(model_path)
        p_tok = Path(tokenizer_dir)

        if p_model.is_file() and p_tok.is_dir() and (p_tok / "vocab.json").is_file():
            _config = ModelConfig(**torch.load(str(p_model), map_location="cpu")["config"]["model"])
            _model = GPT(_config).to("cpu")
            _model.eval()
            _tokenizer = load_tokenizer(str(p_tok))
        else:
            # Fallback in-memory demo model (e.g. for container sanity tests / offline dev)
            import tempfile
            from tokenizers import ByteLevelBPETokenizer
            from minigpt_llm.tokenizer.train_bpe import SPECIAL_TOKENS

            tmp = tempfile.mkdtemp()
            corpus_path = Path(tmp) / "corpus.txt"
            corpus_path.write_text(
                "Once upon a time there was a little GPT model.\n"
                "Hello world! The model is alive and ready to generate text.\n"
                "This is a demo checkpoint running in CPU mode.\n"
            )
            raw_tok = ByteLevelBPETokenizer()
            raw_tok.train([str(corpus_path)], vocab_size=256, min_frequency=1, special_tokens=SPECIAL_TOKENS)
            raw_tok.save_model(tmp)
            _tokenizer = load_tokenizer(tmp)

            _config = ModelConfig(
                vocab_size=256,
                num_layers=2,
                hidden_size=64,
                num_heads=2,
                max_position_embeddings=128,
            )
            _model = GPT(_config).to("cpu")
            _model.eval()

        # --- warm-up pass (prevents first-request latency spike) ---
        with torch.no_grad():
            dummy_input = _tokenizer.encode(" ").ids
            _ = _model.forward(torch.tensor([dummy_input], dtype=torch.long), use_cache=True)


def get_model() -> torch.nn.Module:
    """Return the loaded ``GPT`` model."""
    if _model is None:
        raise RuntimeError("Model not initialized. Call serving.loader.initialize() first.")
    return _model


def get_tokenizer() -> Any:
    """Return the loaded tokenizer."""
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not initialized. Call serving.loader.initialize() first.")
    return _tokenizer


def get_config() -> ModelConfig:
    """Return the model config."""
    if _config is None:
        raise RuntimeError("Config not initialized. Call serving.loader.initialize() first.")
    return _config