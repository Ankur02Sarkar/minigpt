"""Tests for atomic checkpoint save/load and pruning."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from minigpt_llm.model.config import ModelConfig
from minigpt_llm.model.model import GPT
from training.checkpoint import load_checkpoint, save_checkpoint
from training.optimizer import build_optimizer
from training.scheduler import WarmupCosineScheduler


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    cfg = ModelConfig(
        vocab_size=32,
        num_layers=1,
        hidden_size=32,
        num_heads=4,
        max_position_embeddings=16,
    )
    model = GPT(cfg)
    opt = build_optimizer(model, lr=1e-3)
    sched = WarmupCosineScheduler(opt, peak_lr=1e-3, warmup_steps=2, max_steps=20, min_lr_ratio=0.1)
    sched.step()
    path = save_checkpoint(
        tmp_path,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=None,
        step=1,
        best_val_loss=3.0,
        config={"x": 1},
        is_best=True,
    )
    assert path.is_file()
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "best.pt").is_file()

    model2 = GPT(cfg)
    opt2 = build_optimizer(model2, lr=1e-3)
    sched2 = WarmupCosineScheduler(
        opt2, peak_lr=1e-3, warmup_steps=2, max_steps=20, min_lr_ratio=0.1
    )
    payload = load_checkpoint(
        tmp_path / "latest.pt", model=model2, optimizer=opt2, scheduler=sched2
    )
    assert payload["step"] == 1
    for p1, p2 in zip(model.parameters(), model2.parameters(), strict=True):
        assert torch.allclose(p1, p2)


def test_checkpoint_prunes_old(tmp_path: Path) -> None:
    cfg = ModelConfig(
        vocab_size=32,
        num_layers=1,
        hidden_size=32,
        num_heads=4,
        max_position_embeddings=16,
    )
    model = GPT(cfg)
    opt = build_optimizer(model, lr=1e-3)
    sched = WarmupCosineScheduler(opt, peak_lr=1e-3, warmup_steps=1, max_steps=10, min_lr_ratio=0.1)
    for step in range(1, 6):
        save_checkpoint(
            tmp_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=None,
            step=step,
            best_val_loss=1.0,
            config={},
            keep_last=3,
        )
    steps = sorted(tmp_path.glob("step_*.pt"))
    assert len(steps) == 3
    assert {p.name for p in steps} == {"step_3.pt", "step_4.pt", "step_5.pt"}
