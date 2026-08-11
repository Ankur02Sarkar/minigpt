# minigpt-llm

> A from-scratch GPT trained on Microsoft Azure with a single Tesla T4 GPU. Served via OpenAI- and Ollama-compatible APIs. Built to fit inside a $200 Azure free-trial credit.

---

## Quickstart

> **Note:** The project is in **Phase 1 (Data Pipeline)**. Library + unit tests are in-repo; **real datasets only run on Azure B2ms** (`/data`). The model is not trained yet.

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
| Phase 1 — Data Pipeline | 10 | 0* | 🔄 In progress (code ready; Azure run pending) |
| Phase 2 — Model Architecture | 8 | 0 | ⏳ Pending |
| Phase 3 — Training Engine | 9 | 0 | ⏳ Pending |
| Phase 4 — Training Runs | 7 | 0 | ⏳ Pending |
| Phase 5 — Inference & Generation | 3 | 0 | ⏳ Pending |
| Phase 6 — Serving (OpenAI + Ollama) | 7 | 0 | ⏳ Pending |
| Phase 7 — Packaging & OSS Polish | 10 | 0 | ⏳ Pending |
| **Total** | **62** | **8** | |

\*Phase 1 library/CLIs/tests are implemented on branch `phase/1-data-pipeline`. Minor phases flip to ✅ only after the B2ms pipeline produces shards and (for 1.10) the T4 is attached.

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

### Phase 1 — Data Pipeline (🔄 In progress)

| # | Task | Code | Azure | Notes |
|---|---|---|---|---|
| 1.1 | TinyStories raw download | ✅ | ⏳ | `minigpt_llm.data.download` |
| 1.2 | WikiText-103 raw download | ✅ | ⏳ | same |
| 1.3 | FineWeb-Edu streaming scaffold | ✅ | ⏳ | generator, no raw persist |
| 1.4 | FineWeb filter + 500M token cap | ✅ | ⏳ | **runs after BPE train** (see order below) |
| 1.5 | Clean + dedupe | ✅ | ⏳ | SHA-1 doc dedupe → `all_text.txt` |
| 1.6 | BPE tokenizer (32k) | ✅ | ⏳ | `minigpt_llm.tokenizer` |
| 1.7 | Tokenize → `.bin` shards + `meta.pkl` | ✅ | ⏳ | int32 little-endian |
| 1.8 | WikiText 95/5 → `val.bin` | ✅ | ⏳ | token-level split |
| 1.9 | Data-loader smoke test | ✅ | ⏳ | `python -m training.dataset` |
| 1.10 | Move data disk to T4 | 📄 runbook | ⏳ | `docs/PHASE1_RUNBOOK.md` |

**Runtime order on B2ms** (tokenizer must exist before FineWeb encode):

1. Download TinyStories + WikiText  
2. Clean + dedupe  
3. Train BPE  
4. Tokenize TinyStories + WikiText + val split  
5. Stream FineWeb → `fineweb.bin` (500M cap)  
6. Smoke test  

```bash
# azure-prep
python -m scripts.run_phase1_pipeline --data-root /data
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
| Prep VM | `minigpt-prep` (Standard_B2ms, zone 1, ~$0.083/hr) — IP: 172.178.112.133 ✅ |
| Train VM | `minigpt-train` (Standard_NC8as_T4_v3, ~$0.752/hr) — not yet created (Phase 1.10) |
| Data disk | `minigpt-data` (64 GB Standard SSD, zone 1, portable) — attached to prep VM ✅ |
| SSH key | `~/.ssh/id_ed25519` (ed25519, no passphrase) |
| SSH config | `Host azure-prep` → 172.178.112.133 |
| Public IP (for NSG) | `101.0.63.207` + `49.37.169.202` (ports 22, 8080, 11434) |
| Quota | NCASv3_T4: 0/8 ✓, BS Family: 0/10 ✓, Total Regional vCPUs: 0/18 ✓ |
| Budget target | $35-50 (within $200 free-trial credit) |
| Docker base image | `minigpt/base:latest` (9.03 GB, torch 2.13.0+cu130) ✅ |

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
