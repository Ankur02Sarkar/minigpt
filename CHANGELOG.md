# Changelog

All notable changes to `minigpt_llm` will be documented in this file.

## Version 0.1.0 (unreleased)

### Added
- **Phase 0**: Azure bootstrap — resource group, data disk, B2ms/T4 VMs, Docker base image
- **Phase 1**: Data pipeline — TinyStories, WikiText-103, FineWeb-Edu (500M token cap), BPE 32k vocab, token shards, train/val split
- **Phase 2**: Model architecture — GPT from scratch, RoPE, SwiGLU MLP, 13M/26M params, type-configurable via YAML
- **Phase 3**: Training engine — memory-mapped shards, AdamW, WarmupCosineLR, checkpointing, AMP/FP16, eval
- **Phase 4**: Training runs — minigpt-low (50k steps, ~4h T4), minigpt-high (100k steps, ~78.6h T4); 4.8 diagnosis (4 root causes fixed); 4.9 deferred on budget
- **Phase 5**: Inference & generation — KV-cache decode, streaming, stop_strings, interactive chat REPL (slash commands), sampling diagnostics (entropy, n-gram repeat detection, on_token callback)
- **Phase 6**: Serving infrastructure — shared model loader, OpenAI-compatible `/v1/*` routes, Ollama-compatible `/api/*` routes, unified FastAPI app with /health /ready probes, optional Bearer auth (`MINIGPT_API_KEY`), per-IP rate limit (60/min)
- **Phase 7**: Packaging & OSS polish — Production Dockerfile (multi-stage), docker-compose for local dev, MODEL_CARD.md, CONTRIBUTING.md, LICENSE, CHANGELOG.md, README updates, examples notebooks, OSS hygiene (ISSUE/PR templates, CODEOWNERS, SECURITY.md)

### Changed
- `README.md` — updated progress tracker, About Me JSON, infrastructure summary, badges, quickstart serve command
- `AGENTS.md` — all phase markers updated; 6.6 → `[~]`, 6.7 → `[~]` (test validation pending)
- `configs/minigpt-{low,high}.yaml` — probe-2 hparams promoted (dropout 0.0, lr 1e-3, wd 0.05, warmup 10000)

### Fixed
- `tests/test_inference_diagnostics.py` — ruff violations, proper torch skip guard
- `MultiShardDataset` token-weighted inverse-CDF selection (was uniform sampling → ~33% WikiText over-sampling)
- `worker_init_fn` re-seeding dataset `_rng` Generators (was overlapping worker windows → ~½ diversity loss)
- Eval `max_batches` hardcoded → configurable `TrainingConfig.eval_max_batches` (default None = full val; 50-batch was 3.5% of val.bin)
- Added `split_tinystories_val` + `DataPaths.tinystories_val_bin` (enables in-domain low-model target)
- Serving middleware: optional auth + rate limit; `MINIGPT_API_KEY` guard

### Deprecated
- Phase 4.9 full 100k retrain with promoted hparams — deferred on budget ($15→$60 corrected); reproducible from preserved shards

## Version 0.0.1 (2024-01-01)
- Initial project bootstrap
- Azure resource provisioning
- Data pipeline scaffold
