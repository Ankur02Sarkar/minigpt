# minigpt-llm

> A from-scratch GPT trained on Microsoft Azure with a single Tesla T4 GPU. Served via OpenAI- and Ollama-compatible APIs. Built to fit inside a $200 Azure free-trial credit.

---

## Quickstart

> **Note:** **Phase 1–2 complete.** Data shards on Azure `/data/tokenized/`; GPT model (RoPE + SwiGLU) in-repo. Next: Phase 3 training engine. The model is not trained yet.

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
# python -c "from minigpt_llm.model import GPT, load_config; print(GPT(load_config('configs/tiny.yaml')).num_parameters())"

# Train (Phase 3+)
# python -m training.train --config configs/tiny.yaml --data-dir /data/tokenized

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
| Phase 3 — Training Engine | 9 | 0 | ⏳ Pending (next) |
| Phase 4 — Training Runs | 7 | 0 | ⏳ Pending |
| Phase 5 — Inference & Generation | 3 | 0 | ⏳ Pending |
| Phase 6 — Serving (OpenAI + Ollama) | 7 | 0 | ⏳ Pending |
| Phase 7 — Packaging & OSS Polish | 10 | 0 | ⏳ Pending |
| **Total** | **62** | **26** | |

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
| `configs/tiny.yaml` | 6L, 256-dim, 4 heads, ctx 512 | **13,012,224** |
| `configs/medium.yaml` | 8L, 384-dim, 6 heads, ctx 1024 | **26,450,304** |

```bash
from minigpt_llm.model import GPT, load_config
model = GPT(load_config("configs/medium.yaml"))
print(model.num_parameters())  # 26450304
```

```bash
pytest -q tests/model tests/test_overfit.py
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
