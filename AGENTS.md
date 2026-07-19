# AGENTS.md — Operating Manual for `minigpt_llm`

This file is the **source of truth for how work gets done** on this project: every agent, contributor, or automated tool that touches the repo should follow it. It is derived from `ARCHITECTURE.md` and breaks the end-to-end plan into **major phases → minor phases → atomic tasks**. Nothing from the architecture is omitted; this document is exhaustive by design.

> The project is large. The repo is small. Keep this file **big and exact** so the next agent (human or AI) can pick up cold and ship forward without re-deriving the plan.

---

## 0. Global Rules (apply to every phase)

1. **Oracle Cloud is the only machine that touches data.** No dataset is downloaded, stored, cleaned, or tokenized on the MacBook. The Mac is a thin client: editor + `git` + `ssh`.
2. **200 GB storage ceiling is a hard constraint** (100 GB boot + 100 GB block = `/data`). Any plan that pushes past 200 GB must be cut, not stretched.
3. **Streaming is the default for large corpora.** FineWeb-Edu is never mirrored raw; it is iterated via `datasets.load_dataset(..., streaming=True)` and tokenized in-flight.
4. **Phase gates are mandatory.** Do not start phase *N+1* until phase *N* has a green checkpoint, a logged metric, and a short note in the PR description.
5. **Every code change lands via a feature branch + PR.** No direct pushes to `main`. PR title follows Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `test:`).
6. **No silent regressions.** If a training metric moves by > 5% vs. the previous best run, document why in `logs/RUN_NOTES.md` and link the offending commit.
7. **Secrets never enter the repo.** Oracle API keys, HF tokens, Docker Hub creds live in `.env` (gitignored) or Oracle Vault.
8. **`README.md` is a living artifact.** See §0.1 — it must stay current at all times.
9. **Reproducibility first.** Every training run has a `configs/<run>.yaml`, a recorded git SHA, a seed (`SEED=42`), and a `logs/<run>/` directory that contains the full stdout + TensorBoard events.
10. **Defensive defaults.** Every script must guard against missing files, empty shards, OOM, and network dropouts. Fail loudly, never silently.

### 0.1 `README.md` maintenance contract (mandatory)

`README.md` is the public face of this project for **end users and OSS contributors** (people who have never read `ARCHITECTURE.md` or this file). It must be updated **in the same commit as any change that affects**:

- **Setup / install** — new system packages, Python deps, Docker base images, Oracle shape changes.
- **Usage** — any new CLI flag, env var, or config field on `train.py`, `generate.py`, the FastAPI server.
- **API surface** — any new/changed/removed OpenAI or Ollama endpoint, request/response shape, or example `curl`.
- **Data** — new dataset added, streaming policy change, tokenizer vocab change.
- **Model** — new config preset, parameter count change, context-length change.
- **Deployment** — new Docker tag, new env var, new port, new volume mount.
- **Roadmap / status** — every completed phase is checked off; every new in-flight task is added.
- **Breaking changes** — called out under a `## Breaking Changes` section near the top until the next minor release.

When updating `README.md`, follow the user's preferred README style: `img.shields.io` badges, `## About Me`-style persona block in JS-object format inside Markdown, and "AI Engineer + Full-Stack" voice. Keep examples runnable — every `curl` block in the README must be copy-pasteable against a freshly built image.

If you are an agent ending a turn where you changed any of the above and `README.md` is stale, **your turn is not done** — update it first, then commit.

---

## 1. Repository Conventions

- **Language:** Python 3.12+. Type hints required on all public functions. `from __future__ import annotations` at the top of every module.
- **Formatting:** `ruff format` + `ruff check` (line length 100). `mypy --strict` on `model/`, `training/`, `serving/`.
- **Testing:** `pytest`. Every module ships with `tests/test_<module>.py`. Coverage gate: ≥ 70% on `model/`, `training/`, `serving/`.
- **Pre-commit:** `ruff`, `ruff-format`, `mypy`, `pytest -x`. Install via `pre-commit install` after clone.
- **Logging:** `structlog` everywhere. No `print()` in production code.
- **Config:** YAML for model + training, `.env` for secrets, CLI flags override both.
- **Imports:** absolute (`from minigpt_llm.model.gpt import GPT`), never relative.
- **File headers:** one-line docstring describing module purpose, then `__all__`.
- **Branch naming:** `phase/<n>-<slug>` (e.g. `phase/2-attention`), `fix/<slug>`, `chore/<slug>`.
- **Commit messages:** Conventional Commits, scope required (`feat(training): add cosine LR scheduler`).

---

## 2. Phase Map (Major → Minor)

The project is divided into **7 major phases** and **~40 minor phases**. Status markers: `[ ]` pending, `[~]` in progress, `[x]` done, `[!]` blocked.

### Phase 0 — Foundation & Infra (Oracle bootstrap)
*Goal: a reproducible, on-Oracle dev environment with `/data` mounted and the repo running in Docker.*

- **0.1 Provision Oracle VM.Standard.A1.Flex instance** `[ ]`
  - 4 OCPUs / 24 GB RAM, Ubuntu 22.04 LTS minimal.
  - Generate + upload SSH key; disable password auth.
  - Configure Oracle VCN + security list (open 22, 8080, 11434 only to your IP).
- **0.2 Attach 100 GB block volume and mount at `/data`** `[ ]`
  - `iscsiadm` discovery → login → `fdisk` → `mkfs.ext4` → `fstab` entry with `nofail`.
  - Verify with `df -h /data` and a `fio` smoke test (≥ 75 IOPS read, ≥ 50 IOPS write).
- **0.3 Install system packages on the instance** `[ ]`
  - `git`, `build-essential`, `python3.12`, `python3.12-venv`, `docker.io`, `docker-compose-plugin`, `htop`, `tmux`, `unzip`.
  - Add the `ubuntu` user to the `docker` group; `newgrp docker`.
- **0.4 Clone the repo and pin Python env** `[ ]`
  - `git clone git@github.com:<org>/minigpt_llm.git /opt/minigpt_llm`.
  - `python3.12 -m venv .venv && source .venv/bin/activate`.
  - `pip install -U pip uv && uv pip install -r requirements.txt -r requirements-dev.txt`.
- **0.5 Build the base Docker image** `[ ]`
  - `docker build -f docker/Dockerfile.base -t minigpt/base:latest .`.
  - Tag with the git SHA: `docker tag minigpt/base:latest minigpt/base:$SHA`.
- **0.6 Pre-commit + CI bootstrap** `[ ]`
  - `.pre-commit-config.yaml`, `.github/workflows/ci.yml` (lint + typecheck + test on PR).
  - Branch protection on `main`: require CI green + 1 review.
- **0.7 README "Quickstart" skeleton** `[ ]`
  - Badges (CI, license, python, docker), 1-paragraph elevator pitch, 5-line "what is this", 1 `curl` example.
  - Filled out properly in Phase 7 once endpoints exist.

**Exit criteria:** `docker run --rm minigpt/base:latest python -c "import torch; print(torch.__version__)"` succeeds on the instance, and `df -h /data` shows the block volume mounted.

---

### Phase 1 — Data Pipeline (Oracle-only)
*Goal: tokenized, deduplicated `.bin` shards on `/data/tokenized/` ready for training, total ≤ 50 GB.*

- **1.1 TinyStories raw download** `[ ]`
  - `datasets.load_dataset("roneneldan/TinyStories", split="train")` → `/data/raw/tinystories.txt`.
  - Verify SHA-256 and line count (`wc -l` ≈ 2.1M).
  - Disk check: ~7.6 GB.
- **1.2 WikiText-103 raw download** `[ ]`
  - `datasets.load_dataset("wikitext", "wikitext-103-v1", split="train")` → `/data/raw/wikitext103.txt`.
  - Verify ~1.8M lines, ~0.55 GB.
- **1.3 FineWeb-Edu streaming scaffold** `[ ]`
  - `datasets.load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)`.
  - Wrap in a generator that yields `(doc_id, text)`; **no persistence of raw docs**.
  - Add a unit test that asserts the generator advances without buffering more than N docs in memory.
- **1.4 FineWeb-Edu filter + on-the-fly tokenization** `[ ]`
  - Drop docs < 200 chars or > 100k chars.
  - Apply lang-detect sanity (must be `en`, score ≥ 0.9).
  - Tokenize in mini-batches of 1000 docs, append int32 IDs to `/data/tokenized/fineweb.bin`.
  - Stop the stream when the shard hits the configured cap (default: 20 GB / ~2.5B tokens).
- **1.5 Cleaning + dedupe pass** `[ ]`
  - For TinyStories + WikiText: `regex` strip control chars, collapse whitespace, drop empty lines.
  - Concat to `/data/cleaned/all_text.txt` (~8 GB).
  - Dedupe at the doc level: SHA-1 of normalized text → keep first occurrence. Log dedupe ratio.
- **1.6 BPE tokenizer training (vocab 32k)** `[ ]`
  - `tokenizers.ByteLevelBPETokenizer.train(files=[...], vocab_size=32000, min_frequency=2)`.
  - Save `vocab.json` + `merges.txt` to **both** `/data/vocab/` (block) and `/opt/minigpt_llm/tokenizer/` (boot).
  - Verify round-trip encode/decode on a 1k-doc sample.
- **1.7 Tokenize all corpora → `.bin` shards** `[ ]`
  - Read each cleaned text file, encode with the trained tokenizer, write as int32 array.
  - Shards: `tinystories.bin`, `wikitext.bin`, `fineweb.bin` (already done in 1.4).
  - Produce `meta.pkl` with `{shard: {"tokens": int, "dtype": "int32"}}`.
- **1.8 Train/val split** `[ ]`
  - 95/5 split on WikiText only (held-out validation).
  - Save `val.bin` to `/data/tokenized/val.bin`.
- **1.9 Data-loader smoke test** `[ ]`
  - `python -m training.dataset --shard /data/tokenized/tinystories.bin --context 1024 --batch 4`.
  - Assert shapes `(4, 1024)` for `input_ids` and `labels`.
  - Assert no `nan` and that `labels` are `input_ids` shifted by one.

**Exit criteria:** `ls -lah /data/tokenized/` shows all shards + `meta.pkl`; `df -h /data` shows ≤ 60 GB used; smoke test passes.

---

### Phase 2 — Model Architecture
*Goal: a from-scratch GPT implementation in `model/` that matches `configs/*.yaml`, type-checked and unit-tested.*

- **2.1 Config system (`model/config.py`)** `[ ]`
  - `@dataclass(frozen=True)` `ModelConfig` with `vocab_size`, `num_layers`, `hidden_size`, `num_heads`, `max_position_embeddings`, `dropout`, `rope_theta`, `tie_weights`.
  - YAML loader: `load_config("configs/tiny.yaml")` returns a typed `ModelConfig`.
  - Param-count helper: `estimate_params(config) -> int` (verified vs. real instantiation to within 1%).
- **2.2 Token + positional embeddings** `[ ]`
  - `nn.Embedding(vocab_size, hidden_size)` for tokens.
  - **RoPE** (rotary) — no learned positional embeddings. Easier length generalization.
  - Unit test: RoPE on identity returns identity up to FP tolerance.
- **2.3 Causal multi-head self-attention** `[ ]`
  - `q, k, v` projections, split into heads, apply RoPE, scaled dot-product, causal mask, out projection.
  - Use `F.scaled_dot_product_attention` (PyTorch 2.x) when `is_training=False` for speed; explicit mask in train mode.
  - Unit test: mask is lower-triangular; output shape `(B, T, hidden)`.
- **2.4 SwiGLU MLP** `[ ]`
  - `down(act(gate(x)) * up(x))` with hidden dim `≈ 8/3 * hidden_size`, rounded to multiple of 64.
  - Unit test: parameter count matches the canonical LLaMA formula.
- **2.5 Decoder block** `[ ]`
  - Pre-norm, attention + residual, pre-norm, MLP + residual.
  - Configurable dropout. Unit test: residual stream norm stays bounded.
- **2.6 Full GPT model (`model/model.py`)** `[ ]`
  - Embed → N × DecoderBlock → final RMSNorm → tied LM head.
  - `forward(input_ids, labels=None)` returns `CausalLMOutputWithPast(loss, logits)`.
  - `generate(input_ids, max_new_tokens, temperature, top_k, top_p)` with KV-cache.
- **2.7 Weight init + dtype policy** `[ ]`
  - Truncated normal init (std 0.02) for linears, normal init for embeddings.
  - `model.to(dtype=torch.bfloat16)` for A10 paths; FP32 on CPU.
- **2.8 Sanity overfit on a single batch** `[ ]`
  - Generate 1 batch of 1024 tokens, run 200 steps, assert loss < 2.0.
  - Add as `tests/test_overfit.py` so it runs in CI on a CPU-only runner.

**Exit criteria:** `pytest tests/model -q` green; overfit test passes; `configs/large.yaml` instantiates a ~50M-param model.

---

### Phase 3 — Training Engine
*Goal: a CLI `train.py` that reads `.bin` shards, runs mixed-precision training, evaluates on val, and writes checkpoints.*

- **3.1 Memory-mapped dataset (`training/dataset.py`)** `[ ]`
  - `np.memmap` the `.bin` shards; sample random `(context+1)` windows.
  - Multi-worker `DataLoader` with deterministic seeding.
- **3.2 Optimizer (`training/optimizer.py`)** `[ ]`
  - AdamW with decoupled weight decay; β1=0.9, β2=0.95, weight_decay=0.1.
  - No-decay groups for biases + norm weights.
- **3.3 LR scheduler (`training/scheduler.py`)** `[ ]`
  - Linear warmup (default 2000 steps) → cosine decay to 10% of peak.
  - Pure-PyTorch (no `transformers` dependency in core).
- **3.4 Checkpointing (`training/checkpoint.py`)** `[ ]`
  - Save `{model, optimizer, scheduler, scaler, step, config, rng_state}` as a single `.pt` to `/opt/minigpt_llm/checkpoints/`.
  - Atomic write (`*.tmp` → rename) to survive interruption.
  - Keep last 3 + best (by val loss).
- **3.5 Evaluation (`training/evaluate.py`)** `[ ]`
  - Run val loss over `val.bin` in fixed batches; return `{val_loss, val_ppl}`.
  - Optional perplexity-per-domain if multiple val shards exist.
- **3.6 Logging (`training/logging.py`)** `[ ]`
  - `tensorboard` writer under `/opt/minigpt_llm/logs/<run>/`.
  - Stdout via `structlog`; every step logs `loss`, `lr`, `throughput (tok/s)`, `gpu_mem`.
- **3.7 Gradient accumulation + AMP** `[ ]`
  - Configurable `grad_accum`; effective batch = `per_device_batch * grad_accum * world_size`.
  - `torch.amp.autocast("cuda", dtype=bfloat16)` on A10; FP32 on CPU.
  - NaN/Inf guard: skip step, log, do not update.
- **3.8 CLI entry point (`training/train.py`)** `[ ]`
  - `argparse` + `pydantic` validation of CLI args.
  - Required: `--config`, `--data-dir`, `--out-dir`. Optional: `--resume`, `--seed`, `--max-steps`, `--eval-every`.
  - `python -m training.train --config configs/tiny.yaml --data-dir /data/tokenized`.
- **3.9 Smoke training run** `[ ]`
  - 100 steps on the tiny model + TinyStories only.
  - Assert loss decreases monotonically over the first 50 steps.
  - Save `runs/smoke/` artifacts to `checkpoints/` and `logs/`.

**Exit criteria:** `python -m training.train --config configs/tiny.yaml --max-steps 1000` completes, logs to TensorBoard, writes a checkpoint, and `evaluate.py` runs end-to-end on `val.bin`.

---

### Phase 4 — Training Runs (progressive scaling)
*Goal: three production-quality models, fully logged, ready to serve.*

- **4.1 Phase-runner script (`scripts/run_phase.sh`)** `[ ]`
  - Wraps `train.py` with per-phase `max_steps`, eval cadence, and checkpoint retention.
  - Writes `logs/<phase>/RUN.md` with start/end time, peak GPU mem, final loss/PPL.
- **4.2 Phase 1 — 8M model on TinyStories (CPU OK, A10 fast)** `[ ]`
  - Config: `tiny.yaml` (6L, 256-dim, 4 heads, ctx 512).
  - 50k steps, eval every 1k, target val PPL ≤ 25.
  - Documented in `logs/phase1/RUN.md`.
- **4.3 Phase 2 — 20M model on TinyStories + WikiText (A10)** `[ ]`
  - Config: `medium.yaml` (8L, 384-dim, 6 heads, ctx 1024).
  - 100k steps, eval every 2k, target val PPL ≤ 18.
- **4.4 Hyperparam tuning pass** `[ ]`
  - Sweep LR in {3e-4, 5e-4, 7e-4}, warmup in {1k, 2k, 4k} on Phase 2.
  - Pick best, freeze, promote to `configs/large.yaml`.
- **4.5 Phase 3 — 50M model on Tiny + Wiki + FineWeb-Edu stream (A10)** `[ ]`
  - Config: `large.yaml` (12L, 512-dim, 8 heads, ctx 1024).
  - 200k steps, eval every 2k, target val PPL ≤ 12.
  - Run started on A10 (pay-go), resumable from `checkpoints/`.
- **4.6 Long-run monitoring** `[ ]`
  - `tmux`/`screen` session named `train-<phase>`.
  - Cron every 5 min: `nvidia-smi`, `df -h /data`, last log line → `logs/heartbeat.log`.
  - Auto-snapshot checkpoint to Oracle Object Storage every 4h (free egress allowance).
- **4.7 Final eval report** `[ ]`
  - `docs/EVAL.md` with side-by-side loss/PPL curves for all three phases.
  - Sample generations on a fixed prompt set (10 prompts × 3 seeds) → `docs/SAMPLES.md`.

**Exit criteria:** Phase 3 checkpoint under `checkpoints/phase3/best.pt`, val PPL ≤ 12, `EVAL.md` + `SAMPLES.md` published.

---

### Phase 5 — Inference & Generation
*Goal: a clean Python API for prompting the trained model with KV-cache, streaming, and stopping criteria.*

- **5.1 Generation core (`inference/generate.py`)** `[ ]`
  - `generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, stop_strings, stream)`.
  - KV-cache; respects `stop_strings` and `eos_token_id`.
  - Generator that yields tokens (stream=True) or returns the full string.
- **5.2 Interactive chat loop (`inference/chat.py`)** `[ ]`
  - Readline-based REPL; system prompt configurable; multi-turn history in memory.
  - `/reset`, `/system`, `/temp`, `/tokens` slash commands.
- **5.3 Sampling diagnostics** `[ ]`
  - Log per-token top-5 distribution entropy for the first 20 generations.
  - Detect repetition loops (n-gram repeat in last 64 tokens) and warn.

**Exit criteria:** `python -m inference.chat --checkpoint checkpoints/phase3/best.pt` produces coherent multi-turn output in < 200ms/token on A10.

---

### Phase 6 — Serving (OpenAI + Ollama APIs)
*Goal: a FastAPI server that is a drop-in replacement for both OpenAI's `/v1/*` and Ollama's `/api/*` endpoints.*

- **6.1 Shared model loader (`serving/loader.py`)** `[ ]`
  - Lazy-load on first request; hold in module-level singleton; thread-safe.
  - Warm-up pass on load to avoid first-request latency spike.
- **6.2 OpenAI-compatible routes (`serving/app_openai.py`)** `[ ]`
  - `POST /v1/chat/completions` — supports `stream=true` (SSE) and `stream=false`.
  - `POST /v1/completions` — legacy single-string prompt.
  - `GET /v1/models` — returns `{data: [{id, owned_by, created}]}`.
  - `POST /v1/embeddings` — pooled last-token embedding (if hidden_size matches a registered encoder).
  - Pydantic request/response models mirroring OpenAI's schema byte-for-byte.
- **6.3 Ollama-compatible routes (`serving/app_ollama.py`)** `[ ]`
  - `POST /api/generate` — NDJSON streaming.
  - `POST /api/chat` — NDJSON streaming.
  - `GET /api/tags` — list local models.
  - `GET /api/version` — return server version.
  - `POST /api/show` — return model info.
- **6.4 Unified server entry (`serving/server.py`)** `[ ]`
  - Mounts both routers on a single FastAPI app; `/health` and `/ready` for k8s-style probes.
  - `uvicorn serving.server:app --host 0.0.0.0 --port 8080`.
  - `--model` flag chooses the checkpoint; `--api openai|ollama|both`.
- **6.5 Auth + rate limiting** `[ ]`
  - Optional `Authorization: Bearer <key>` checked against `.env`-defined keys.
  - Per-IP token bucket; default 60 req/min, configurable.
- **6.6 Streaming correctness tests** `[ ]`
  - SSE frames parse as expected; NDJSON frames are line-delimited.
  - `stop` reason is one of `stop|length|content_filter`.
- **6.7 Cross-client compatibility** `[ ]`
  - Verify with `openai-python` SDK against `/v1/chat/completions`.
  - Verify with the official Ollama CLI against `/api/chat` and `/api/generate`.
  - Pin client versions in `tests/integration/requirements.txt`.

**Exit criteria:** `openai.ChatCompletion.create(model="minigpt", messages=[...])` returns a valid response; `ollama run minigpt "Hello"` streams a response.

---

### Phase 7 — Packaging, Distribution & OSS Polish
*Goal: anyone with Docker can run the model; anyone with a browser can read the README and know exactly what to do.*

- **7.1 Production Dockerfile (`docker/Dockerfile`)** `[ ]`
  - Multi-stage: build (with dev deps, compile wheels) → runtime (slim base).
  - `HEALTHCHECK` calling `/health`.
  - Labels: `org.opencontainers.image.source`, `image.version`, `image.revision`.
- **7.2 docker-compose for local dev (`docker-compose.yml`)** `[ ]`
  - Service `train` (volume mount `/data`, GPU passthrough), service `serve` (port 8080).
  - `.env.example` with every var documented.
- **7.3 Model card (`MODEL_CARD.md`)** `[ ]`
  - Intended use, out-of-scope use, training data summary, eval results, limitations, safety considerations.
  - Cite the data sources and respect their licenses.
- **7.4 `README.md` — final, public-facing pass** `[ ]`
  - Badges, hero GIF / screenshot, quickstart, install, run, API examples, training, contributing, license, citation.
  - JS-object "About Me" persona block (per user style).
  - Every `curl` example copy-paste tested against the running image.
- **7.5 `CONTRIBUTING.md`** `[ ]`
  - Dev setup, pre-commit, PR template, code-of-conduct reference.
- **7.6 `LICENSE`** `[ ]`
  - Apache-2.0 (matches `img.shields.io` defaults; permissive for OSS).
- **7.7 `CHANGELOG.md`** `[ ]`
  - Keep a Changelog format; one entry per release; semver.
- **7.8 Examples & notebooks (`examples/`)** `[ ]`
  - `examples/quickstart.ipynb` — load checkpoint, generate 100 tokens.
  - `examples/serve_locally.ipynb` — run the Docker image, hit endpoints.
- **7.9 OSS hygiene** `[ ]`
  - `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`.
  - `SECURITY.md` with disclosure policy.
  - `CODEOWNERS` mapping `model/`, `training/`, `serving/` to the maintainer.
- **7.10 First release (v0.1.0)** `[ ]`
  - Tag, GitHub release with notes, Docker image pushed to `ghcr.io/<org>/minigpt_llm:0.1.0`.
  - Cross-post to HN/Reddit (optional).

**Exit criteria:** A stranger can `docker run ghcr.io/<org>/minigpt_llm:0.1.0` and chat with the model within 5 minutes, guided only by `README.md`.

---

## 3. Cross-Phase Concerns (continuous, not a single phase)

- **A. Observability** — every long-running script (`train.py`, `serve.py`, the streaming tokenizers) emits structured logs and (where applicable) Prometheus metrics on `/metrics`.
- **B. Security** — no model checkpoint contains secrets; HF tokens loaded from env; `gitleaks` in CI.
- **C. Cost guardrails** — A10 pay-go spend is capped; an alert at 80% of monthly budget shuts the GPU instance down.
- **D. Data license tracking** — every dataset added is logged in `docs/DATA_LICENSES.md` with version + license + SHA of the loader script.
- **E. Reproducibility matrix** — a single `scripts/repro.sh` that, from a clean clone, reproduces the published checkpoint and metrics. Runs in CI nightly.
- **F. Dependency hygiene** — Dependabot weekly; `uv pip compile requirements.in` → `requirements.txt` is the source of truth.

---

## 4. Long-Run Training & Resilience

*Core principle: training runs on the Oracle VM. The Mac is a remote control — it can sleep, shut down, or be reformatted mid-run without affecting the process. SSH is just a remote terminal; the only thing that kills the run is something happening to the Oracle VM itself (reboot, OOM, disk full, crash). The infrastructure below is designed so that even those events are recoverable without the operator being present.*

### 4.1 Why not tmux (or nohup)?

`tmux` and `nohup ... &` only solve the **SSH-disconnect** problem. They do **nothing** for:

- Oracle reboot (kernel updates, host maintenance)
- OOM kill
- Training crash (NaN loss, segfault, CUDA error)
- Disk full mid-shard
- A bug that crashes the process 3 seconds after start (loops forever)

For any multi-hour run we use **`systemd`**, which is built for "run this forever, restart on failure, resume from where you stopped." tmux/nohup are acceptable only for runs shorter than ~1 hour.

### 4.2 Canonical training systemd unit

File: `/etc/systemd/system/minigpt-train.service`

```ini
[Unit]
Description=minigpt training (phase N)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/opt/minigpt_llm
ExecStart=/opt/minigpt_llm/.venv/bin/python -m training.train \
    --config configs/<run>.yaml \
    --data-dir /data/tokenized \
    --out-dir /opt/minigpt_llm/checkpoints/<run> \
    --resume latest
Restart=on-failure
RestartSec=30
StartLimitBurst=10
StartLimitIntervalSec=600
OOMScoreAdjust=-500
# Refuse to start if /data is > 95% full (we'd just OOM/crash anyway)
ExecStartPre=/bin/sh -c 'df -h /data | awk "NR==2 && $5+0 < 95" | grep -q .'
StandardOutput=journal
StandardError=journal
SyslogIdentifier=minigpt-train
ExecStopPost=/opt/minigpt_llm/scripts/notify_done.sh

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now minigpt-train
```

### 4.3 Observability stack (no terminal required)

| Question | Command (run from your Mac) |
|---|---|
| Is it running? | `ssh oracle 'systemctl is-active minigpt-train'` |
| Last 50 log lines | `ssh oracle 'journalctl -u minigpt-train -n 50 --no-pager'` |
| Current step / loss / GPU / disk | `ssh oracle 'cat /opt/minigpt_llm/STATUS.json'` |
| How many times has it crashed since start? | `ssh oracle 'systemctl show minigpt-train -p NRestarts'` |
| Loss curves in a browser | `ssh -L 6006:localhost:6006 oracle` → `http://localhost:6006` |

### 4.4 Heartbeat (cron, every 5 min)

`/opt/minigpt_llm/scripts/heartbeat.sh`:
```bash
#!/bin/bash
LOG=/opt/minigpt_llm/logs/phase3/run.log
STEP=$(grep -oP 'step=\K\d+' "$LOG" | tail -1)
LOSS=$(grep -oP 'loss=\K[0-9.]+' "$LOG" | tail -1)
GPU=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "n/a")
DISK=$(df -h /data | tail -1 | awk '{print $5}')
cat > /opt/minigpt_llm/STATUS.json <<EOF
{"step":${STEP:-0},"loss":${LOSS:-null},"gpu":"${GPU}","disk":"${DISK}","ts":"$(date -Iseconds)"}
EOF
```

Crontab entry:
```
*/5 * * * * /opt/minigpt_llm/scripts/heartbeat.sh
```

`STATUS.json` is the single source of truth for "is the run healthy?" — readable from any device with SSH.

### 4.5 Push notifications (ntfy.sh — free, no account)

Install the **ntfy** app on your phone (Android/iOS), subscribe to a secret topic, then wire it in:

```bash
WEBHOOK="https://ntfy.sh/<your-secret-topic>"
notify() { curl -s -d "$1" "$WEBHOOK" >/dev/null; }
```

Wire points:
- `ExecStopPost=/opt/minigpt_llm/scripts/notify_done.sh` in the systemd unit → fires on clean exit or systemd restart-then-give-up.
- `trap 'notify "minigpt: training crashed step=${STEP:-?} loss=${LOSS:-?}"' ERR` in the training wrapper → fires on any non-zero exit.
- Heartbeat cron: if `step` hasn't advanced in 6 consecutive runs, fire a "stuck" alert.

Fallback (no phone signal): `echo "minigpt: training done" | mail -s "minigpt" you@gmail.com`.

### 4.6 Failure-mode matrix

| Failure | Without systemd (bare `python ...` over SSH) | With systemd unit |
|---|---|---|
| SSH drops / Mac sleeps | dies (unless tmux/nohup) | survives |
| Mac shut down / reformatted | dies (unless tmux/nohup) | survives |
| Oracle reboots (host maintenance) | dies | auto-restarts after boot, `--resume` from last checkpoint |
| Training crash (OOM, NaN, CUDA error) | dies | auto-restarts after 30s, `--resume` |
| Disk full on `/data` | crashes silently mid-shard | `ExecStartPre` refuses to start, alert fires |
| Kernel OOM-kills the process | dies | `OOMScoreAdjust=-500` makes us less likely to be killed; if we are, systemd restarts cleanly |
| Bug loops (crash within 10s of start, forever) | infinite tight loop | `StartLimitBurst=10` per 600s → systemd gives up, alert fires |
| `fineweb.bin` write gets interrupted | corrupt shard | atomic write + re-validate on next start |

### 4.7 Checkpoint + `--resume` contract

- `train.py --resume latest` always restarts from `checkpoints/<run>/latest.pt`.
- `checkpoints/<run>/` lives on the **boot volume** (not `/data`); survives reboots and re-attachments.
- `latest.pt` is updated atomically (`.tmp` → `rename`) every N steps (default 1000). Power loss mid-write ≠ corrupt checkpoint.
- Best-by-val-loss checkpoint is kept separately as `best.pt`.
- Keep the last 3 `latest` snapshots to recover from a single corrupted write.
- Every 4h: snapshot `best.pt` to Oracle Object Storage (free egress) via `rclone`. Off-host backup of the thing you cannot rebuild.

### 4.8 TensorBoard service (browser-based loss curves)

`/etc/systemd/system/minigpt-tb.service`:
```ini
[Unit]
Description=TensorBoard for minigpt
After=network-online.target

[Service]
Type=simple
User=ubuntu
ExecStart=/opt/minigpt_llm/.venv/bin/tensorboard \
    --logdir /opt/minigpt_llm/logs --host 0.0.0.0 --port 6006
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now minigpt-tb
```

Access from your laptop: `ssh -L 6006:localhost:6006 oracle` → `http://localhost:6006`. Phone-friendly too if you keep the tunnel open in Termius/BlinkShell.

### 4.9 Pre-flight checklist before any run > 1 hour

Before kicking off a long training run, the operator (or agent) MUST verify:

1. A `<run>.service` unit exists at `/etc/systemd/system/minigpt-<run>.service`.
2. `systemctl enable --now minigpt-<run>` returns 0.
3. `systemctl is-active minigpt-<run>` returns `active` after 60s.
4. `cat /opt/minigpt_llm/STATUS.json` updates within 5 min.
5. `journalctl -u minigpt-<run> -n 5` shows expected boot output (model loaded, dataset opened, step 1 logged).
6. ntfy topic is configured and a test notification (`notify "test"`) arrives on the phone.
7. `checkpoints/<run>/` exists and is writable.

If any of these fail, **do not walk away.** Fix first.

### 4.10 The rule for the next agent

Do not start a multi-hour training run inside a plain `ssh oracle && python ...` command. That is a footgun. Always:
1. Write a `<run>.service` unit.
2. `systemctl enable --now <run>.service`.
3. Run the §4.9 pre-flight checklist.
4. Disconnect.

For runs under 1 hour, `tmux` or `nohup ... &` is fine and does not require a unit file. The threshold is "would I be annoyed if this died and I had to restart it from scratch?"

---

## 5. Definition of Done (per minor phase)

A minor phase is "done" only when:

1. Code is merged to `main` via PR with green CI.
2. Tests covering the new behavior are added (unit, integration, or smoke — whichever fits).
3. `logs/<run>/` artifacts are committed (or linked from the PR) for any non-trivial run.
4. `README.md` is updated if any of the §0.1 trigger conditions apply.
5. `ARCHITECTURE.md` is updated if the architecture itself changed (new shape, new dataset, new endpoint).
6. `AGENTS.md` (this file) is updated: flip the `[ ]` → `[x]` and add any newly discovered minor phases discovered during execution.

---

## 6. Handoff Protocol (end of an agent's turn)

Before ending any turn, an agent must:

1. Update todos (the live `TodoWrite` list reflects only current work; this file is the long-term ledger).
2. Flip status markers in §2 for completed minor phases.
3. Run the §5 Definition-of-Done checklist mentally; fix any red items.
4. If blocked, write a one-line `## Blocked on <X>` note at the bottom of this file (keep it; remove when unblocked).
5. Leave a short summary message in chat: "Done: <phases>. Next: <phase>. Blocked: <…>."

---

## 7. Phase Dependency Graph (textual)

```
0 (Foundation)
 └─> 1 (Data)
      └─> 2 (Model)
           └─> 3 (Training Engine)
                └─> 4 (Training Runs)
                     ├─> 5 (Inference)
                     │    └─> 6 (Serving)
                     │         └─> 7 (Packaging/OSS)
                     └─> 7 (Packaging/OSS — docs only)
```

Phases 5 and 7 (docs-only slice) can begin as soon as 4 is in motion, in parallel with 6.

---

*This file is intentionally long. If you find yourself wanting to skip a section, that's the section most likely to bite you later. Read it again.*
