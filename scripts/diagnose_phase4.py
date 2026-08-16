"""Phase 4.8 Step B diagnosis on minigpt-high/best.pt (rules eval harness in/out of fault,
establishes a full-val baseline, and reports vocab usage)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import structlog
import torch

from minigpt_llm.model import GPT, ModelConfig
from training.evaluate import evaluate_shard

__all__ = ["main"]

log = structlog.get_logger(__name__)

_DTYPE = np.dtype("<i4")


def _load_model(checkpoint: Path, device: torch.device) -> GPT:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ModelConfig(**payload["config"]["model"])
    model = GPT(config).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    log.info(
        "model_loaded",
        checkpoint=str(checkpoint),
        step=payload.get("step"),
        best_val_loss=payload.get("best_val_loss"),
    )
    return model


def vocab_usage(shard_path: Path, vocab_size: int) -> dict[str, float | int]:
    data = np.memmap(shard_path, dtype=_DTYPE, mode="r")
    ids = np.unique(data)
    return {
        "vocab_size": vocab_size,
        "unique_ids": int(ids.size),
        "coverage_pct": round(float(ids.size) / vocab_size * 100, 2),
        "min_id": int(ids.min()),
        "max_id": int(ids.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-shard", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    model = _load_model(args.checkpoint, device)

    # 1) harness baseline at 50 batches (matches the original run's logged number)
    m50 = evaluate_shard(
        model,
        args.val_shard,
        context_length=args.context_length,
        batch_size=args.batch_size,
        max_batches=50,
        device=device,
    )
    # 2) full-val (the Phase 4.8 default) — the true baseline
    mfull = evaluate_shard(
        model,
        args.val_shard,
        context_length=args.context_length,
        batch_size=args.batch_size,
        max_batches=None,
        device=device,
    )
    # 3) vocab usage on the val shard
    vocab = vocab_usage(args.val_shard, model.config.vocab_size)

    report = {
        "checkpoint": str(args.checkpoint),
        "val_shard": str(args.val_shard),
        "eval_50_batches": m50,
        "eval_full_val": mfull,
        "vocab_usage": vocab,
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
