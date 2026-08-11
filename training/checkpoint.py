"""Atomic checkpoint save/load with retention of last N step snapshots."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import structlog
import torch
import torch.nn as nn
from torch.optim import Optimizer

from training.scheduler import WarmupCosineScheduler

__all__ = [
    "load_checkpoint",
    "resolve_resume_path",
    "save_checkpoint",
]

log = structlog.get_logger(__name__)

_STEP_RE = re.compile(r"^step_(\d+)\.pt$")


def resolve_resume_path(out_dir: Path, resume: str) -> Path | None:
    """Resolve ``latest`` / ``best`` / path / ``none`` to a checkpoint file."""
    if resume in ("", "none", "None"):
        return None
    if resume == "latest":
        p = out_dir / "latest.pt"
        return p if p.is_file() else None
    if resume == "best":
        p = out_dir / "best.pt"
        return p if p.is_file() else None
    p = Path(resume)
    if not p.is_file():
        raise FileNotFoundError(f"resume checkpoint not found: {p}")
    return p


def save_checkpoint(
    out_dir: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: WarmupCosineScheduler,
    scaler: Any | None,
    step: int,
    best_val_loss: float,
    config: dict[str, Any],
    keep_last: int = 3,
    is_best: bool = False,
) -> Path:
    """Atomically write ``latest.pt`` and ``step_{n}.pt``; optionally ``best.pt``."""
    if "/mnt" in str(out_dir.resolve()):
        raise ValueError(f"refusing to write checkpoints under ephemeral /mnt: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_val_loss": best_val_loss,
        "config": config,
        "rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state()

    step_path = out_dir / f"step_{step}.pt"
    _atomic_torch_save(payload, step_path)
    _atomic_torch_save(payload, out_dir / "latest.pt")
    if is_best:
        _atomic_torch_save(payload, out_dir / "best.pt")

    _prune_step_checkpoints(out_dir, keep_last=keep_last)
    log.info("checkpoint_saved", path=str(step_path), step=step, is_best=is_best)
    return step_path


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: WarmupCosineScheduler | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load checkpoint into modules; returns payload (includes step, best_val_loss)."""
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # torch < 2.4 has no weights_only kwarg
        payload = torch.load(path, map_location=map_location)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and "scheduler" in payload:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if "rng_state" in payload:
        torch.set_rng_state(payload["rng_state"])
    log.info("checkpoint_loaded", path=str(path), step=payload.get("step"))
    return payload


def _atomic_torch_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _prune_step_checkpoints(out_dir: Path, *, keep_last: int) -> None:
    steps: list[tuple[int, Path]] = []
    for p in out_dir.glob("step_*.pt"):
        m = _STEP_RE.match(p.name)
        if m:
            steps.append((int(m.group(1)), p))
    steps.sort(key=lambda x: x[0])
    for _, p in steps[:-keep_last] if keep_last > 0 else steps:
        try:
            p.unlink()
            log.info("checkpoint_pruned", path=str(p))
        except OSError:
            log.warning("checkpoint_prune_failed", path=str(p))
