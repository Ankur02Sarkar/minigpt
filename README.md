# minigpt-llm

> A from-scratch GPT trained on Microsoft Azure with a single Tesla T4 GPU. Served via OpenAI- and Ollama-compatible APIs. Built to fit inside a $200 Azure free-trial credit.

---

## Quickstart

> **Note:** **Phase 1–4 complete.** Both training runs finished (`minigpt-low` → `minigpt-high`) and the Azure T4 is **deallocated** to save credits. See [`docs/EVAL.md`](docs/EVAL.md) for results and [`docs/SAMPLES.md`](docs/SAMPLES.md) for generations. Next: Phase 5 (inference).

```bash
# Clone
git clone https://github.com/Ankur02Sarkar/minigpt.git
cd minigpt

# Set up env
python3.12 -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .

# Unit tests (synthetic data only — no HF download)
pytest -q

# --- on azure-prep only (never on a laptop) ---
# export HF_TOKEN=...
# python -m scripts.run_phase1_pipeline --data-root /data
# python -m training.dataset --shard /data/tokenized/tinystories.bin --context 1024 --batch 4

# Model smoke (Phase 2+)
# python -c "from minigpt_llm.model import GPT, load_config; print(GPT(load_config('configs/minigpt-low.yaml')).num_parameters())"

# Train (Phase 3+) — on azure-train with /data mounted
# python -m training.train \
#   --config configs/minigpt-low.yaml \
#   --data-dir /data/tokenized \
#   --out-dir /opt/minigpt_llm/checkpoints/smoke \
#   --max-steps 1000 --device cuda

# Serve (Phase 6+)
# uvicorn serving.server:app --host 0.0.0.0 --port 8080
```

Full Azure steps: [`docs/PHASE1_RUNBOOK.md`](docs/PHASE1_RUNBOOK.md). Dataset licenses: [`docs/DATA_LICENSES.md`](docs/DATA_LICENSES.md).

---

## About Me

```js
{
  project: "minigpt-llm",
  author: "Ankur Sarkar",
  role: "AI Engineer + Full-Stack",
  goal: "Train a from-scratch GPT on Azure T4 GPU for <$50, serve it with OpenAI + Ollama compatible APIs",
  stack: ["Python 3.12", "PyTorch", "FastAPI", "Azure", "Docker"],
  constraints: {
    budget: "$35-50 (within $200 free-trial credit)",
    hardware: "1× NVIDIA Tesla T4 16GB (Turing architecture, FP16 only — no BF16)",
    deadline: "30 days or credit exhaustion, whichever comes first",
    token_cap: "500M FineWeb-Edu tokens (not 2-3B)",
  },
}
```

---

## Project Status & Progress Tracker

> **For new agent sessions:** Read this section first. It tells you exactly what's done, what's in progress, and what's next. Then read `AGENTS.md` for the full plan.

### Overall Progress

| Major Phase | Minor Phases | Completed | Status |
|---|---|---|---|
| Phase 0 — Foundation & Infra | 8 | 8 | ✅ Done |
| Phase 1 — Data Pipeline | 10 | 10 | ✅ Done |
| Phase 2 — Model Architecture | 8 | 8 | ✅ Done |
| Phase 3 — Training Engine | 9 | 9 | ✅ Done |
| Phase 4 — Training Runs | 9 | 8 | ✅ Trained + diagnosed (4.8 probe-2 wins → configs promoted; 4.9 retrain pending) |
| Phase 5 — Inference & Generation | 3 | 0 | ⏳ Pending |
| Phase 6 — Serving (OpenAI + Ollama) | 7 | 0 | ⏳ Pending |
| Phase 7 — Packaging & OSS Polish | 10 | 0 | ⏳ Pending |
| **Total** | **64** | **48** | |

### Phase 0 — Foundation & Infra (✅ Complete)

| # | Task | Status | Notes |
|---|---|---|---|
| 0.1 | Create Azure resource group + managed data disk | ✅ | `minigpt-rg` in eastus, 64 GB Standard SSD in zone 1 |
| 0.2 | Provision B2ms prep VM | ✅ | `minigpt-prep` in zone 1, SSH key ed25519, NSG: 22/8080/11434 to 49.37.169.202 |
| 0.3 | Format + mount `/data` on B2ms | ✅ | ext4 on /dev/sdc1, 63 GB usable, fstab entry with nofail |
| 0.4 | Install system packages on B2ms | ✅ | git, python3.12 (deadsnakes PPA), docker CE, build-essential, htop, tmux, unzip |
| 0.5 | Clone repo + pin Python env | ✅ | Cloned to /opt/minigpt_llm, uv venv with all deps, torch 2.13.0+cu130 |
| 0.6 | Build Docker base image | ✅ | `minigpt/base:latest` (9.03 GB), tagged `minigpt/base:545a5c2` |
| 0.7 | Pre-commit + CI bootstrap | ✅ | `.pre-commit-config.yaml` + `.github/workflows/ci.yml` created |
| 0.8 | README "Quickstart" skeleton | ✅ | This file |

**Exit criteria verified:**
- `docker run --rm minigpt/base:latest python -c "import torch; print(torch.__version__)"` → `2.13.0+cu130` ✅
- `df -h /data` → 63 GB mounted ✅

### Phase 1 — Data Pipeline (✅ Complete)

| # | Task | Status | Notes |
|---|---|---|---|
| 1.1 | TinyStories raw download | ✅ | ~2.12M lines → tokenized; raw deleted after |
| 1.2 | WikiText-103 raw download | ✅ | `Salesforce/wikitext` (hub 1.x); raw deleted after |
| 1.3 | FineWeb-Edu streaming scaffold | ✅ | no raw FineWeb on disk |
| 1.4 | FineWeb filter + 500M token cap | ✅ | **500M tokens** hit cap → `fineweb.bin` (1.9 GB) |
| 1.5 | Clean + dedupe | ✅ | ~15.6% dedupe ratio; cleaned text deleted after tokenize |
| 1.6 | BPE tokenizer (32k) | ✅ | 100% round-trip; `/data/vocab` + train OS copy |
| 1.7 | Tokenize → `.bin` shards + `meta.pkl` | ✅ | tinystories 445M · wikitext 109M · fineweb 500M |
| 1.8 | WikiText 95/5 → `val.bin` | ✅ | val 5.75M tokens |
| 1.9 | Data-loader smoke test | ✅ | `(4, 1024)` on B2ms + T4 |
| 1.10 | Move data disk to T4 | ✅ | `minigpt-train` NC8as_T4_v3 zone 1; `nvidia-smi` Tesla T4 16 GB |

**Artifacts on data disk (`/data`, ~4 GB used after cleanup):**

| Path | Tokens / notes |
|---|---|
| `/data/tokenized/tinystories.bin` | 444,696,201 |
| `/data/tokenized/wikitext.bin` | 109,330,614 |
| `/data/tokenized/val.bin` | 5,754,243 |
| `/data/tokenized/fineweb.bin` | 500,000,000 |
| `/data/tokenized/meta.pkl` | shard metadata |
| `/data/vocab/{vocab.json,merges.txt}` | BPE 32k |

```bash
# on azure-train (when VM is started)
python -m training.dataset --shard /data/tokenized/tinystories.bin --context 1024 --batch 4
```

### Phase 2 — Model Architecture (✅ Complete)

| # | Task | Status | Notes |
|---|---|---|---|
| 2.1 | Config + YAML | ✅ | `minigpt_llm.model.config`, `configs/{tiny,medium}.yaml` |
| 2.2 | RoPE | ✅ | no learned PE |
| 2.3 | Causal MHA | ✅ | train mask + SDPA eval; KV-cache |
| 2.4 | SwiGLU MLP | ✅ | intermediate ≈ 8/3 H, ×64 |
| 2.5 | Decoder block | ✅ | pre-norm RMSNorm |
| 2.6 | Full GPT + generate | ✅ | tied LM head, top-k/top-p |
| 2.7 | Weight init + dtype | ✅ | trunc-normal; **no BF16** |
| 2.8 | Overfit single batch | ✅ | `tests/test_overfit.py` loss &lt; 2.0 |

**Configs (measured params, vocab 32k, tied head):**

| Config | Spec | Params |
|---|---|---|
| **minigpt-low** (`configs/minigpt-low.yaml`) | 6L, 256-dim, 4 heads, ctx 512 | **13,012,224** |
| **minigpt-high** (`configs/minigpt-high.yaml`) | 8L, 384-dim, 6 heads, ctx 1024 | **26,450,304** |

```bash
from minigpt_llm.model import GPT, load_config
model = GPT(load_config("configs/minigpt-high.yaml"))
print(model.num_parameters())  # 26450304
```

```bash
pytest -q tests/model tests/test_overfit.py
```

### Phase 3 — Training Engine (✅ Complete)

| # | Task | Status | Notes |
|---|---|---|---|
| 3.1 | Memmap multi-shard dataset | ✅ | `MultiShardDataset`, worker seed |
| 3.2 | AdamW optimizer | ✅ | decay / no-decay groups |
| 3.3 | Warmup + cosine LR | ✅ | pure PyTorch |
| 3.4 | Atomic checkpoints | ✅ | latest / best / last 3 steps |
| 3.5 | Val eval | ✅ | loss + perplexity |
| 3.6 | Logging | ✅ | structlog + TensorBoard |
| 3.7 | Grad accum + FP16 AMP | ✅ | CUDA only; FP32 on CPU; no BF16 |
| 3.8 | CLI | ✅ | `python -m training.train` |
| 3.9 | Smoke | ✅ | `tests/test_train_smoke.py` |

```bash
# Production queue (preferred) — survives SSH disconnect
# See docs/PHASE4_RUNBOOK.md
sudo systemctl enable --now minigpt-train-queue
cat /opt/minigpt_llm/STATUS.json
```

Models: **minigpt-low** (50k steps, TinyStories) then **minigpt-high** (100k steps, full corpus).  
Defaults (T4): `per_device_batch=4`, `grad_accum=16` → effective batch 64, FP16 autocast.  
Both `best.pt` checkpoints exist — the T4 has been **deallocated**.

### Phase 4 — Training Runs (✅ Trained + diagnosed — retrain pending)

| Model | Config | Steps | Target val PPL | Actual val PPL | Wall time | best.pt |
|---|---|---|---|---|---|---|
| **minigpt-low** | `minigpt-low.yaml` | 50k | ≤ 25 (invalid*) | ~14,745 | 9.4 h | 149 MB |
| **minigpt-high** | `minigpt-high.yaml` | 100k | ≤ 18 | ~111 | 78.6 h | 303 MB |

\* minigpt-low trained on TinyStories but was evaluated on the WikiText val shard — a cross-domain target it could never hit. Full breakdown, val-loss curves, cost record, and Phase 4.8 diagnosis + probe results in [`docs/EVAL.md`](docs/EVAL.md); verbatim generations in [`docs/SAMPLES.md`](docs/SAMPLES.md).

Both runs finished exit 0 under the systemd queue (`phase: both_complete`), then the T4 was deallocated to preserve credits. Checkpoints persist on the OS disk; `/data` intact.

**Phase 4.8 — PPL diagnosis + tuning probes (2026-08-15/16):** 3 × 5k-step probes on the T4 diagnosed 4 root causes (uniform shard sampling, per-worker RNG not re-seeded, 50-batch eval prefix, no TinyStories val split) — all fixed. Probe results (full-val PPL = e^loss):

| Probe | Config deltas | Train loss @5k | Full-val PPL |
|---|---|---|---|
| Probe-1 (sampler-fix-only) | token-weighted sampler + per-worker reseed | 4.29 | 256.8 |
| **Probe-2 🏆 (hparam)** | + dropout 0.0, lr 1e-3, wd 0.05, warmup 500 | **4.06** | **207.2** |
| Probe-3 (mix) | + wikitext 3× upweight (probe-1 base) | 4.56 | 208.8 |

Probe-2's hparams were **promoted into `configs/minigpt-{low,high}.yaml`** (warmup scaled to 10% of each run). **Next:** task 4.9 — full 100k-step retrain (~20h T4, ~$15) to confirm the extrapolated improvement (~60-80 PPL vs original 111). See `docs/EVAL.md` for full probe analysis.

Ops: [`docs/PHASE4_RUNBOOK.md`](docs/PHASE4_RUNBOOK.md) · systemd unit `minigpt-train-queue` · heartbeat `STATUS.json`.

```bash
# resume work on the models (billing resumes)
az vm start -g minigpt-rg -n minigpt-train
ssh azure-train 'cat /opt/minigpt_llm/STATUS.json; ls /opt/minigpt_llm/checkpoints/minigpt-*/best.pt'
```

### Local repo setup (Track B — ✅ Complete)

| # | Task | Status |
|---|---|---|
| B1 | `.gitignore` expanded | ✅ |
| B2 | Project skeleton created | ✅ |
| B3 | `pyproject.toml`, `requirements*.txt` | ✅ |
| B4 | Local venv with `uv` | ✅ |
| B5 | `.pre-commit-config.yaml` | ✅ |
| B6 | `.github/workflows/ci.yml` | ✅ |
| B7 | `docker/Dockerfile.base` | ✅ |
| B8 | README.md skeleton + progress tracker | ✅ |
| B9 | AGENTS.md `id_rsa` → `id_ed25519` | ✅ |

### Azure environment (Track A — ✅ Complete)

| # | Task | Status |
|---|---|---|
| A1 | Create resource group `minigpt-rg` + 64 GB managed data disk | ✅ |
| A2 | Provision B2ms prep VM + attach disk + NSG rules | ✅ |
| A3 | Format + mount `/data` on B2ms | ✅ |
| A4 | Install system packages on B2ms | ✅ |
| A5 | Clone repo + uv venv on B2ms VM | ✅ |
| A6 | Build Docker base image on B2ms + verify exit criteria | ✅ |

---

## Infrastructure Summary

| Resource | Value |
|---|---|
| Azure subscription | "Azure subscription 1" (free trial, $200 credit) |
| Resource group | `minigpt-rg` (eastus) ✅ |
| Prep VM | `minigpt-prep` (Standard_B2ms, zone 1) — **deallocated** after data prep ✅ |
| Train VM | `minigpt-train` (Standard_NC8as_T4_v3, zone 1, ~$0.752/hr) — IP: 48.217.83.172; **deallocated after Phase 1 verify** (start only to train) ✅ |
| Data disk | `minigpt-data` (64 GB Standard SSD, zone 1) — attached to **train** VM at `/data` ✅ |
| SSH key | `~/.ssh/id_ed25519` (ed25519, no passphrase) |
| SSH config | `Host azure-train` → 48.217.83.172 · `Host azure-prep` (deallocated) |
| Public IP (for NSG) | `101.0.63.207` + `49.37.169.202` (ports 22, 8080, 11434) |
| GPU | Tesla T4 16 GB, driver 610.57, torch 2.11.0+cu128 ✅ |
| Budget target | $35-50 (within $200 free-trial credit) |
| Docker base image | `minigpt/base:latest` (9.03 GB) ✅ |

---

## Architecture

See `ARCHITECTURE.md` (gitignored — local only) for the full design document. The high-level flow:

```
Data (FineWeb-Edu + TinyStories + WikiText-103)
  → BPE tokenizer (32k vocab)
  → Tokenized .bin shards on Azure managed disk
  → GPT model (8M → 20M params, RoPE + SwiGLU)
  → FP16 training on T4 (AdamW + cosine LR)
  → Checkpoint
  → FastAPI server (OpenAI + Ollama compatible)
  → Docker container
```

---

## License

Apache-2.0 (planned — Phase 7.6)
