"""Sanity overfit: single batch should drive loss below 2.0 on CPU."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.config import ModelConfig
from minigpt_llm.model.model import GPT


def test_overfit_single_batch() -> None:
    """2-layer mini model, one fixed batch, 200 AdamW steps, loss < 2.0."""
    torch.manual_seed(42)
    cfg = ModelConfig(
        vocab_size=64,
        num_layers=2,
        hidden_size=64,
        num_heads=4,
        max_position_embeddings=128,
        dropout=0.0,
        tie_weights=True,
    )
    model = GPT(cfg)
    model.train()

    # Fixed synthetic batch (B=2, T=64) — fits CI CPU budget
    batch = torch.randint(0, cfg.vocab_size, (2, 64))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))

    losses: list[float] = []
    for _step in range(200):
        opt.zero_grad(set_to_none=True)
        out = model(batch, labels=batch)
        assert out.loss is not None
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.item()))

    assert losses[0] > losses[-1], f"loss did not decrease: {losses[0]:.3f} → {losses[-1]:.3f}"
    assert losses[-1] < 2.0, f"final loss {losses[-1]:.3f} not < 2.0"
