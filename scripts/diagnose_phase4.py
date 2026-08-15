"""Phase 4.8 Step B diagnosis on minigpt-high/best.pt (rules eval harness in/out of fault,
establishes a full-val baseline, and reports vocab usage)."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import structlog
import torch
import torch.nn.functional as F

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


def manual_ce(
    model: GPT,
    shard_path: Path,
    *,
    context_length: int,
    batch_size: int,
    n_batches: int,
    device: torch.device,
) -> dict[str, float]:
    """Independent CE loss pass (recomputed from logits) to cross-check evaluate_shard."""
    data = np.memmap(shard_path, dtype=_DTYPE, mode="r")
    n = int(data.shape[0])
    window = context_length + 1
    starts = list(range(0, n - window + 1, context_length))
    total_loss = 0.0
    total = 0
    nb = 0
    idx = 0
    with torch.no_grad():
        while idx < len(starts) and nb < n_batches:
            bx: list[torch.Tensor] = []
            by: list[torch.Tensor] = []
            for _ in range(batch_size):
                if idx >= len(starts):
                    break
                s = starts[idx]
                idx += 1
                chunk = np.array(data[s : s + window], dtype=np.int64)
                bx.append(torch.from_numpy(chunk[:-1].copy()))
                by.append(torch.from_numpy(chunk[1:].copy()))
            if not bx:
                break
            x = torch.stack(bx).to(device)
            y = torch.stack(by).to(device)
            logits = model(x)
            sl = logits[..., :-1, :].reshape(-1, logits.size(-1)).float()
            tl = y[..., 1:].reshape(-1)
            loss = F.cross_entropy(sl, tl)
            tok = x.size(0) * context_length
            total_loss += float(loss.item()) * tok
            total += tok
            nb += 1
    avg = total_loss / max(total, 1)
    return {
        "manual_loss": avg,
        "manual_ppl": math.exp(min(avg, 100.0)),
        "manual_tokens": float(total),
    }


def vocab_usage(shard_path: Path, vocab_size: int) -> dict[str, float | int]:
    """Fraction of the vocab actually present in the val shard (token-fertility proxy)."""
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
    # 3) independent manual CE cross-check (50 batches)
    manual = manual_ce(
        model,
        args.val_shard,
        context_length=args.context_length,
        batch_size=args.batch_size,
        n_batches=50,
        device=device,
    )
    # 4) vocab usage on the val shard
    vocab = vocab_usage(args.val_shard, model.config.vocab_size)

    report = {
        "checkpoint": str(args.checkpoint),
        "val_shard": str(args.val_shard),
        "eval_50_batches": m50,
        "eval_full_val": mfull,
        "manual_ce_50": manual,
        "harness_matches_manual": abs(m50["val_loss"] - manual["manual_loss"]) < 0.05,
        "vocab_usage": vocab,
    }
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
