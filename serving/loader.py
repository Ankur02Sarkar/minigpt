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
    """
    global _model, _tokenizer, _config

    with _class_lock:
        if _model is not None:
            return  # already initialized

        _config = ModelConfig(**torch.load(model_path, map_location="cpu")["config"]["model"])
        _model = GPT(_config).to("cpu")
        _model.eval()
        _tokenizer = load_tokenizer(tokenizer_dir)

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