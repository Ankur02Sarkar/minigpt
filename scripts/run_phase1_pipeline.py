#!/usr/bin/env python3
"""Orchestrate Phase 1 data pipeline on the Azure B2ms prep VM.

Implementation order (tokenizer must exist before FineWeb tokenization)::

    1.1 TinyStories download
    1.2 WikiText-103 download
    1.5 Clean + dedupe
    1.6 Train BPE (32k)
    1.7 Tokenize TinyStories + WikiText
    1.8 Val split (WikiText 95/5)
    1.3–1.4 Stream FineWeb-Edu → fineweb.bin (500M token cap)
    1.9 Dataset smoke test

Usage (on azure-prep)::

    export HF_TOKEN=...
    python -m scripts.run_phase1_pipeline --data-root /data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import structlog

from minigpt_llm.data.clean import clean_and_dedupe_files
from minigpt_llm.data.download import download_tinystories, download_wikitext
from minigpt_llm.data.fineweb import FINEWEB_TOKEN_CAP, stream_and_tokenize_fineweb
from minigpt_llm.data.paths import DataPaths
from minigpt_llm.data.shards import (
    load_meta,
    save_meta,
    split_wikitext_val,
    tokenize_text_file,
)
from minigpt_llm.tokenizer.load import eos_token_id, load_tokenizer, make_encode_batch
from minigpt_llm.tokenizer.train_bpe import train_bpe
from training.dataset import build_batch

__all__ = ["main", "run_pipeline"]

log = structlog.get_logger(__name__)


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


def _skip_if_exists(path: Path, force: bool, step: str) -> bool:
    if path.exists() and not force:
        log.info("step_skip_exists", step=step, path=str(path))
        return True
    return False


def run_pipeline(
    data_root: Path,
    *,
    force: bool = False,
    skip_fineweb: bool = False,
    fineweb_token_cap: int = FINEWEB_TOKEN_CAP,
    smoke_context: int = 1024,
    smoke_batch: int = 4,
    seed: int = 42,
    steps: list[str] | None = None,
) -> dict[str, Any]:
    """Run selected pipeline steps. ``steps=None`` runs all."""
    paths = DataPaths(data_root)
    paths.ensure_dirs()
    all_steps = [
        "download_tinystories",
        "download_wikitext",
        "clean",
        "train_bpe",
        "tokenize_corpora",
        "val_split",
        "fineweb",
        "smoke",
    ]
    selected = set(steps) if steps else set(all_steps)
    results: dict[str, Any] = {"data_root": str(paths.root)}

    if "download_tinystories" in selected:
        if _skip_if_exists(paths.tinystories_raw, force, "download_tinystories"):
            results["download_tinystories"] = {"skipped": True}
        else:
            results["download_tinystories"] = download_tinystories(paths, force=force)

    if "download_wikitext" in selected:
        if _skip_if_exists(paths.wikitext_raw, force, "download_wikitext"):
            results["download_wikitext"] = {"skipped": True}
        else:
            results["download_wikitext"] = download_wikitext(paths, force=force)

    if "clean" in selected:
        if _skip_if_exists(paths.all_text_cleaned, force, "clean"):
            results["clean"] = {"skipped": True}
        else:
            results["clean"] = clean_and_dedupe_files(paths, force=force)

    if "train_bpe" in selected:
        vocab_json = paths.vocab / "vocab.json"
        if _skip_if_exists(vocab_json, force, "train_bpe"):
            results["train_bpe"] = {"skipped": True}
        else:
            results["train_bpe"] = train_bpe(paths, force=force)

    if "tokenize_corpora" in selected:
        tok = load_tokenizer(paths)
        encode_fn = make_encode_batch(tok)
        eos = eos_token_id(tok)
        meta = load_meta(paths.meta_pkl)

        if not _skip_if_exists(paths.tinystories_bin, force, "tokenize_tinystories"):
            # Tokenize from cleaned all_text is mixed; re-clean per raw file for pure shards.
            # Prefer per-corpus cleaned: re-read raw with same clean_line for each corpus.
            n = _tokenize_raw_corpus(
                paths.tinystories_raw,
                paths.tinystories_bin,
                encode_fn,
                force=force,
                eos_id=eos,
            )
            meta["tinystories"] = {"tokens": n, "dtype": "int32"}
            results["tokenize_tinystories"] = {"tokens": n}
        else:
            results["tokenize_tinystories"] = {"skipped": True}

        if not _skip_if_exists(paths.wikitext_bin, force, "tokenize_wikitext"):
            n = _tokenize_raw_corpus(
                paths.wikitext_raw,
                paths.wikitext_bin,
                encode_fn,
                force=force,
                eos_id=eos,
            )
            meta["wikitext"] = {"tokens": n, "dtype": "int32"}
            results["tokenize_wikitext"] = {"tokens": n}
        else:
            results["tokenize_wikitext"] = {"skipped": True}

        save_meta(paths.meta_pkl, meta)

    if "val_split" in selected:
        if paths.val_bin.exists() and not force:
            log.info("step_skip_exists", step="val_split", path=str(paths.val_bin))
            results["val_split"] = {"skipped": True}
        else:
            results["val_split"] = split_wikitext_val(paths, force=force)

    if "fineweb" in selected and not skip_fineweb:
        if _skip_if_exists(paths.fineweb_bin, force, "fineweb"):
            results["fineweb"] = {"skipped": True}
        else:
            tok = load_tokenizer(paths)
            encode_fn = make_encode_batch(tok)
            eos = eos_token_id(tok)
            results["fineweb"] = stream_and_tokenize_fineweb(
                paths,
                encode_fn,
                token_cap=fineweb_token_cap,
                force=force,
                eos_id=eos,
            )
    elif "fineweb" in selected and skip_fineweb:
        log.info("fineweb_skipped_by_flag")
        results["fineweb"] = {"skipped": True, "reason": "skip_fineweb"}

    if "smoke" in selected:
        smoke_shard = paths.tinystories_bin
        if not smoke_shard.is_file():
            # Fall back to wikitext if tinystories not present
            smoke_shard = paths.wikitext_bin
        if not smoke_shard.is_file():
            raise FileNotFoundError("no shard available for smoke test")
        batch = build_batch(
            smoke_shard,
            context=smoke_context,
            batch=smoke_batch,
            seed=seed,
        )
        assert batch["input_ids"].shape == (smoke_batch, smoke_context)
        assert batch["labels"].shape == (smoke_batch, smoke_context)
        results["smoke"] = {
            "shard": str(smoke_shard),
            "input_shape": list(batch["input_ids"].shape),
            "labels_shape": list(batch["labels"].shape),
            "ok": True,
        }
        log.info("smoke_ok", **results["smoke"])

    return results


def _tokenize_raw_corpus(
    raw_path: Path,
    out_bin: Path,
    encode_fn: Any,
    *,
    force: bool,
    eos_id: int | None,
) -> int:
    """Clean lines on the fly from a raw one-doc-per-line file and tokenize."""
    from minigpt_llm.data.clean import clean_line

    if not raw_path.is_file():
        raise FileNotFoundError(f"missing raw corpus: {raw_path}")
    # Write a temporary cleaned side file next to out for reuse, or stream via temp.
    # Use a sidecar cleaned file under cleaned/ named after stem.
    cleaned_side = raw_path.parent.parent / "cleaned" / f"{raw_path.stem}.cleaned.txt"
    cleaned_side.parent.mkdir(parents=True, exist_ok=True)
    if not cleaned_side.exists() or force:
        tmp = cleaned_side.with_suffix(cleaned_side.suffix + ".tmp")
        n = 0
        with (
            raw_path.open("r", encoding="utf-8", errors="replace") as fin,
            tmp.open("w", encoding="utf-8") as fout,
        ):
            for line in fin:
                c = clean_line(line)
                if c:
                    fout.write(c)
                    fout.write("\n")
                    n += 1
        tmp.replace(cleaned_side)
        log.info("per_corpus_cleaned", path=str(cleaned_side), lines=n)

    return tokenize_text_file(
        cleaned_side,
        out_bin,
        encode_fn,
        force=force,
        add_eos=eos_id is not None,
        eos_id=eos_id,
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = argparse.ArgumentParser(description="Phase 1 data pipeline (Azure B2ms)")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Absolute path to data disk root (e.g. /data). Required — no Mac default.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts")
    parser.add_argument(
        "--skip-fineweb",
        action="store_true",
        help="Skip FineWeb stream (useful for dry runs)",
    )
    parser.add_argument(
        "--fineweb-token-cap",
        type=int,
        default=FINEWEB_TOKEN_CAP,
        help=f"Hard token cap for FineWeb (default {FINEWEB_TOKEN_CAP})",
    )
    parser.add_argument("--smoke-context", type=int, default=1024)
    parser.add_argument("--smoke-batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[
            "download_tinystories",
            "download_wikitext",
            "clean",
            "train_bpe",
            "tokenize_corpora",
            "val_split",
            "fineweb",
            "smoke",
        ],
        help="Run only these steps",
    )
    args = parser.parse_args(argv)

    root = args.data_root.expanduser()
    if not root.is_absolute():
        log.error("data_root_must_be_absolute", path=str(root))
        return 2

    try:
        results = run_pipeline(
            root,
            force=args.force,
            skip_fineweb=args.skip_fineweb,
            fineweb_token_cap=args.fineweb_token_cap,
            smoke_context=args.smoke_context,
            smoke_batch=args.smoke_batch,
            seed=args.seed,
            steps=args.only,
        )
    except Exception:
        log.exception("pipeline_failed")
        return 1

    log.info("pipeline_done", steps=list(results.keys()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
