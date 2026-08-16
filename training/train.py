"""Training CLI: AMP FP16 (CUDA) / FP32 (CPU), grad accum, eval, checkpoints."""

from __future__ import annotations

import argparse
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from pydantic import BaseModel, field_validator

from minigpt_llm.model import GPT
from minigpt_llm.model.init import assert_not_bfloat16
from training.checkpoint import load_checkpoint, resolve_resume_path, save_checkpoint
from training.config import TrainingConfig, load_train_config
from training.dataset import build_dataloader, infinite_loader
from training.evaluate import evaluate_shard
from training.logging import TrainLogger, configure_structlog
from training.optimizer import build_optimizer
from training.scheduler import WarmupCosineScheduler

__all__ = ["TrainArgs", "main", "run_training"]

log = structlog.get_logger(__name__)


class TrainArgs(BaseModel):
    """Validated CLI arguments (overrides YAML where set)."""

    config: Path
    data_dir: Path
    out_dir: Path
    resume: str = "none"
    seed: int | None = None
    max_steps: int | None = None
    eval_every: int | None = None
    lr: float | None = None
    device: str = "auto"
    per_device_batch: int | None = None
    grad_accum: int | None = None
    eval_max_batches: int | None = None

    @field_validator("config", "data_dir", "out_dir", mode="before")
    @classmethod
    def _path(cls, v: Any) -> Path:
        return Path(v)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if spec == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device: {spec}")


def _apply_cli_overrides(train_cfg: TrainingConfig, args: TrainArgs) -> TrainingConfig:
    data = train_cfg.to_dict()
    if args.seed is not None:
        data["seed"] = args.seed
    if args.max_steps is not None:
        data["max_steps"] = args.max_steps
    if args.eval_every is not None:
        data["eval_every"] = args.eval_every
    if args.lr is not None:
        data["lr"] = args.lr
    if args.per_device_batch is not None:
        data["per_device_batch"] = args.per_device_batch
    if args.grad_accum is not None:
        data["grad_accum"] = args.grad_accum
    if args.eval_max_batches is not None:
        data["eval_max_batches"] = args.eval_max_batches
    return TrainingConfig(**data)


def run_training(args: TrainArgs) -> dict[str, Any]:
    """Execute training; returns summary metrics."""
    configure_structlog()
    if str(args.out_dir.resolve()).startswith("/mnt") or "/mnt/" in str(args.out_dir):
        raise ValueError("out_dir must not be on ephemeral /mnt")

    model_cfg, train_cfg = load_train_config(args.config)
    train_cfg = _apply_cli_overrides(train_cfg, args)
    device = _resolve_device(args.device)
    use_cuda = device.type == "cuda"
    _set_seed(train_cfg.seed)

    data_dir = args.data_dir
    train_paths = [data_dir / name for name in train_cfg.train_shards]
    for p in train_paths:
        if not p.is_file():
            raise FileNotFoundError(f"train shard missing: {p}")
    val_path = data_dir / train_cfg.val_shard
    if not val_path.is_file():
        log.warning("val_shard_missing", path=str(val_path))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    logger = TrainLogger(args.out_dir)

    model = GPT(model_cfg).to(device)
    assert_not_bfloat16(next(model.parameters()).dtype)
    opt = build_optimizer(
        model,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        beta1=train_cfg.beta1,
        beta2=train_cfg.beta2,
    )
    sched = WarmupCosineScheduler(
        opt,
        peak_lr=train_cfg.lr,
        warmup_steps=train_cfg.warmup_steps,
        max_steps=train_cfg.max_steps,
        min_lr_ratio=train_cfg.min_lr_ratio,
    )
    # Prefer torch.amp.GradScaler when available (2.x); fall back for older builds.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

    start_step = 0
    best_val = float("inf")
    resume_path = resolve_resume_path(args.out_dir, args.resume)
    if resume_path is not None:
        payload = load_checkpoint(
            resume_path,
            model=model,
            optimizer=opt,
            scheduler=sched,
            scaler=scaler if use_cuda else None,
            map_location=device,
        )
        start_step = int(payload.get("step", 0))
        best_val = float(payload.get("best_val_loss", best_val))

    loader = build_dataloader(
        train_paths,
        context_length=train_cfg.context_length,
        batch_size=train_cfg.per_device_batch,
        num_workers=train_cfg.num_workers if use_cuda else 0,
        seed=train_cfg.seed,
        pin_memory=use_cuda,
    )
    batches = infinite_loader(loader)

    config_blob = {
        "model": {
            "vocab_size": model_cfg.vocab_size,
            "num_layers": model_cfg.num_layers,
            "hidden_size": model_cfg.hidden_size,
            "num_heads": model_cfg.num_heads,
            "max_position_embeddings": model_cfg.max_position_embeddings,
            "dropout": model_cfg.dropout,
            "rope_theta": model_cfg.rope_theta,
            "tie_weights": model_cfg.tie_weights,
        },
        "training": train_cfg.to_dict(),
    }

    log.info(
        "train_start",
        device=str(device),
        params=model.num_parameters(),
        max_steps=train_cfg.max_steps,
        start_step=start_step,
        effective_batch=train_cfg.effective_batch,
        amp=use_cuda,
    )

    model.train()
    step = start_step
    tokens_per_step = train_cfg.per_device_batch * train_cfg.grad_accum * train_cfg.context_length

    while step < train_cfg.max_steps:
        t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        skipped = False

        for _micro in range(train_cfg.grad_accum):
            batch = next(batches)
            input_ids = batch["input_ids"].to(device, non_blocking=use_cuda)
            labels = batch["labels"].to(device, non_blocking=use_cuda)
            ctx = (
                torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                if use_cuda
                else nullcontext()
            )
            with ctx:
                out = model(input_ids, labels=labels)
                if out.loss is None:
                    raise RuntimeError("model returned no loss")
                loss = out.loss / train_cfg.grad_accum
            if use_cuda:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            loss_accum += float(out.loss.detach().item())

        # NaN/Inf guard
        if not math_isfinite(loss_accum):
            log.warning("non_finite_loss_skip", step=step, loss=loss_accum)
            opt.zero_grad(set_to_none=True)
            if use_cuda:
                scaler.update()
            skipped = True
        else:
            if use_cuda:
                scaler.unscale_(opt)
            if train_cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            if use_cuda:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

        lr = sched.step()
        step += 1
        dt = time.perf_counter() - t0
        tok_s = tokens_per_step / max(dt, 1e-9)

        if not skipped and step % train_cfg.log_every == 0:
            gpu_mem = 0.0
            if use_cuda:
                gpu_mem = torch.cuda.max_memory_allocated() / (1024**3)
            logger.log_train(
                step,
                {
                    "loss": loss_accum / train_cfg.grad_accum,
                    "lr": lr,
                    "tok_s": tok_s,
                    "gpu_mem_gb": gpu_mem,
                },
            )

        if val_path.is_file() and step % train_cfg.eval_every == 0:
            metrics = evaluate_shard(
                model,
                val_path,
                context_length=train_cfg.context_length,
                batch_size=train_cfg.per_device_batch,
                max_batches=train_cfg.eval_max_batches,
                device=device,
            )
            logger.log_val(step, metrics)
            is_best = metrics["val_loss"] < best_val
            if is_best:
                best_val = metrics["val_loss"]
            save_checkpoint(
                args.out_dir,
                model=model,
                optimizer=opt,
                scheduler=sched,
                scaler=scaler if use_cuda else None,
                step=step,
                best_val_loss=best_val,
                config=config_blob,
                is_best=is_best,
            )
            model.train()
        elif step % train_cfg.ckpt_every == 0:
            save_checkpoint(
                args.out_dir,
                model=model,
                optimizer=opt,
                scheduler=sched,
                scaler=scaler if use_cuda else None,
                step=step,
                best_val_loss=best_val,
                config=config_blob,
                is_best=False,
            )

    # final checkpoint
    save_checkpoint(
        args.out_dir,
        model=model,
        optimizer=opt,
        scheduler=sched,
        scaler=scaler if use_cuda else None,
        step=step,
        best_val_loss=best_val,
        config=config_blob,
        is_best=False,
    )
    logger.close()
    summary = {"final_step": step, "best_val_loss": best_val}
    log.info("train_done", **summary)
    return summary


def math_isfinite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="minigpt training engine")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume", type=str, default="none")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--per-device-batch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument(
        "--eval-max-batches",
        type=int,
        default=None,
        help="cap eval to N batches (default: full val shard)",
    )
    ns = parser.parse_args(argv)

    try:
        args = TrainArgs(
            config=ns.config,
            data_dir=ns.data_dir,
            out_dir=ns.out_dir,
            resume=ns.resume,
            seed=ns.seed,
            max_steps=ns.max_steps,
            eval_every=ns.eval_every,
            lr=ns.lr,
            device=ns.device,
            per_device_batch=ns.per_device_batch,
            grad_accum=ns.grad_accum,
            eval_max_batches=ns.eval_max_batches,
        )
        run_training(args)
    except Exception:
        configure_structlog()
        log.exception("train_failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
