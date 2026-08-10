# minigpt-llm

> A from-scratch GPT trained on Microsoft Azure with a single Tesla T4 GPU. Served via OpenAI- and Ollama-compatible APIs. Built to fit inside a $200 Azure free-trial credit.

---

## Quickstart

> **Note:** The project is currently in **Phase 0 (Foundation & Infra)**. The model is not trained yet. End-to-end quickstart will be available after Phase 7.

```bash
# Clone
git clone https://github.com/Ankur02Sarkar/minigpt.git
cd minigpt

# Set up env
python3.12 -m venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt

# Train (Phase 3+)
python -m training.train --config configs/tiny.yaml --data-dir /data/tokenized

# Serve (Phase 6+)
uvicorn serving.server:app --host 0.0.0.0 --port 8080
```

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
| Phase 0 — Foundation & Infra | 8 | 0 | 🔄 In Progress |
| Phase 1 — Data Pipeline | 10 | 0 | ⏳ Pending |
| Phase 2 — Model Architecture | 8 | 0 | ⏳ Pending |
| Phase 3 — Training Engine | 9 | 0 | ⏳ Pending |
| Phase 4 — Training Runs | 7 | 0 | ⏳ Pending |
| Phase 5 — Inference & Generation | 3 | 0 | ⏳ Pending |
| Phase 6 — Serving (OpenAI + Ollama) | 7 | 0 | ⏳ Pending |
| Phase 7 — Packaging & OSS Polish | 10 | 0 | ⏳ Pending |
| **Total** | **62** | **0** | |

### Phase 0 — Foundation & Infra (Current)

| # | Task | Status | Notes |
|---|---|---|---|
| 0.1 | Create Azure resource group + managed data disk | ⏳ | Next: `az group create`, `az disk create` |
| 0.2 | Provision B2ms prep VM | ⏳ | SSH key ready (`~/.ssh/id_ed25519`), quota approved (BS: 0/10, NCASv3_T4: 0/8, Total vCPU: 0/18) |
| 0.3 | Format + mount `/data` on B2ms | ⏳ | |
| 0.4 | Install system packages on B2ms | ⏳ | git, python3.12, docker, htop, tmux, unzip |
| 0.5 | Clone repo + pin Python env | ⏳ | Repo skeleton created locally; needs push + clone on VM |
| 0.6 | Build Docker base image | ⏳ | `docker/Dockerfile.base` created |
| 0.7 | Pre-commit + CI bootstrap | ⏳ | `.pre-commit-config.yaml` + `.github/workflows/ci.yml` created |
| 0.8 | README "Quickstart" skeleton | ✅ | This file |

### Local repo setup (Track B — partial Phase 0 work done locally)

| # | Task | Status |
|---|---|---|
| B1 | `.gitignore` expanded | ✅ |
| B2 | Project skeleton created (`minigpt_llm/`, `training/`, `serving/`, `inference/`, `configs/`, `docker/`, `scripts/`, `tests/`, `docs/`, `.github/`) | ✅ |
| B3 | `pyproject.toml`, `requirements.txt`, `requirements-dev.txt` created | ✅ |
| B4 | Local venv with `uv` | ⏳ |
| B5 | `.pre-commit-config.yaml` created | ✅ |
| B6 | `.github/workflows/ci.yml` created | ✅ |
| B7 | `docker/Dockerfile.base` created | ✅ |
| B8 | README.md skeleton + progress tracker | ✅ |
| B9 | AGENTS.md `id_rsa` → `id_ed25519` | ⏳ |

### Azure environment (Track A — not yet started)

| # | Task | Status |
|---|---|---|
| A1 | Create resource group `minigpt-rg` + 64 GB managed data disk | ⏳ |
| A2 | Provision B2ms prep VM + attach disk + NSG rules | ⏳ |
| A3 | Format + mount `/data` on B2ms | ⏳ |
| A4 | Install system packages on B2ms | ⏳ |
| A5 | Clone repo + uv venv on B2ms VM | ⏳ |
| A6 | Build Docker base image on B2ms + verify exit criteria | ⏳ |

---

## Infrastructure Summary

| Resource | Value |
|---|---|
| Azure subscription | "Azure subscription 1" (free trial, $200 credit) |
| Resource group | `minigpt-rg` (eastus) — not yet created |
| Prep VM | `minigpt-prep` (Standard_B2ms, ~$0.083/hr) — not yet created |
| Train VM | `minigpt-train` (Standard_NC8as_T4_v3, ~$0.752/hr) — not yet created |
| Data disk | `minigpt-data` (64 GB Standard SSD, portable between VMs) — not yet created |
| SSH key | `~/.ssh/id_ed25519` (ed25519, no passphrase) |
| Public IP (for NSG) | `49.37.169.202` |
| Quota | NCASv3_T4: 0/8 ✓, BS Family: 0/10 ✓, Total Regional vCPUs: 0/18 ✓ |
| Budget target | $35-50 (within $200 free-trial credit) |

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
