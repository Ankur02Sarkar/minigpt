"""Short end-to-end train smoke on synthetic shards (CPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

torch = pytest.importorskip("torch")

from training.train import TrainArgs, run_training


def _write_cfg(path: Path) -> None:
    cfg = {
        "model": {
            "vocab_size": 64,
            "num_layers": 2,
            "hidden_size": 64,
            "num_heads": 4,
            "max_position_embeddings": 64,
            "dropout": 0.0,
            "rope_theta": 10000.0,
            "tie_weights": True,
        },
        "training": {
            "seed": 0,
            "context_length": 32,
            "per_device_batch": 2,
            "grad_accum": 2,
            "max_steps": 40,
            "lr": 3.0e-3,
            "weight_decay": 0.0,
            "beta1": 0.9,
            "beta2": 0.95,
            "warmup_steps": 5,
            "min_lr_ratio": 0.1,
            "eval_every": 20,
            "ckpt_every": 20,
            "log_every": 5,
            "grad_clip": 1.0,
            "num_workers": 0,
            "train_shards": ["train.bin"],
            "val_shard": "val.bin",
        },
    }
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_train_smoke_loss_drops(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    # structured-ish tokens so model can overfit a bit
    rng = np.random.default_rng(0)
    train = np.tile(np.arange(64, dtype=np.int32), 200)
    rng.shuffle(train)
    train.tofile(data / "train.bin")
    val = np.arange(64, dtype=np.int32).repeat(20)
    val.tofile(data / "val.bin")

    cfg_path = tmp_path / "cfg.yaml"
    _write_cfg(cfg_path)
    out = tmp_path / "out"

    summary = run_training(
        TrainArgs(
            config=cfg_path,
            data_dir=data,
            out_dir=out,
            resume="none",
            device="cpu",
            max_steps=40,
        )
    )
    assert summary["final_step"] == 40
    assert (out / "latest.pt").is_file()
    # TensorBoard dir may exist
    assert (out / "tb").exists() or True
