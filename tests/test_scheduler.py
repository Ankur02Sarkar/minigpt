"""Tests for warmup cosine LR schedule."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.scheduler import WarmupCosineScheduler


def test_warmup_and_cosine_floor() -> None:
    m = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = WarmupCosineScheduler(
        opt,
        peak_lr=1e-3,
        warmup_steps=10,
        max_steps=110,
        min_lr_ratio=0.1,
    )
    lrs = [sched.step() for _ in range(110)]
    assert lrs[0] == pytest.approx(1e-3 / 10, rel=1e-5)
    assert lrs[9] == pytest.approx(1e-3, rel=1e-5)
    # after warmup, decreasing trend overall
    assert lrs[20] > lrs[100]
    # last step is max_steps-1 → near floor (exact floor at step >= max_steps)
    assert lrs[-1] == pytest.approx(1e-4, rel=1e-2)
    assert sched.get_lr(110) == pytest.approx(1e-4, rel=1e-5)


def test_scheduler_state_dict_roundtrip() -> None:
    m = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = WarmupCosineScheduler(opt, peak_lr=1e-3, warmup_steps=5, max_steps=50, min_lr_ratio=0.1)
    for _ in range(12):
        sched.step()
    state = sched.state_dict()
    m2 = torch.nn.Linear(2, 2)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    sched2 = WarmupCosineScheduler(
        opt2, peak_lr=1e-3, warmup_steps=5, max_steps=50, min_lr_ratio=0.1
    )
    sched2.load_state_dict(state)
    assert sched2.last_step == 11
    assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
