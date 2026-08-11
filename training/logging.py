"""Structlog + TensorBoard helpers for training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

__all__ = ["TrainLogger", "configure_structlog"]


def configure_structlog() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


class TrainLogger:
    """Thin wrapper: structured stdout + optional TensorBoard scalars."""

    def __init__(self, log_dir: Path) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._log = structlog.get_logger("train")
        self._writer: Any = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
        except Exception:
            self._log.warning("tensorboard_unavailable")

    def log_train(self, step: int, metrics: dict[str, float]) -> None:
        self._log.info("train_step", step=step, **metrics)
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(f"train/{k}", v, step)

    def log_val(self, step: int, metrics: dict[str, float]) -> None:
        self._log.info("val_step", step=step, **metrics)
        if self._writer is not None:
            for k, v in metrics.items():
                self._writer.add_scalar(f"val/{k}", v, step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
