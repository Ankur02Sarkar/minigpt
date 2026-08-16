# Model Card — minigpt_llm

## 📋 Intended Use
`minigpt_llm` is a GPT-style language model trained from scratch on a combination of TinyStories, WikiText-103, and FineWeb-Edu (500M token cap). It is intended for:
- Next-token completion and chat-style generation
- Research into transformer architecture, training dynamics, and efficient serving
- Educational experimentation with from-scratch GPT implementations

## 🚫 Out-of-Scope Use
- Generation of personally identifiable information (PII) beyond training data scope
- Disallowed or harmful content generation
- Military or surveillance applications
- Real-time interactive systems requiring low-latency safety guarantees
- Tool use, function calling, or agentic behavior beyond next-token prediction

## 📊 Training Data Summary
| Dataset | Size | Notes |
|---|---|---|
| TinyStories | 2,119,489 lines ~1.8 GB | Erik Schulz; deduplicated |
| WikiText-103 | 1,165,029 non-empty lines ~511 MB | Salesforce/wikitext v1; deduplicated |
| FineWeb-Edu | 500,000,000 token cap ~1.9 GB | Streaming; length/lang filter; 470k docs kept |

**Total tokens:** ~1.06B (TinyStories + WikiText + FineWeb)  
**Deduplication ratio:** ~15.6% (across all corpora)  
**Preprocessing:** Cleaning + dedupe pass; tokenized with 32k BPE vocab

## 📈 Eval Results
| Config | Params | Context | Steps | Val PPL (on held-out) |
|---|---|---|---|---|
| minigpt-low | ~13M | 512 | 50k | ~14.7k (on TinyStories val) |
| minigpt-high | ~26M | 1024 | 100k | ~111 (on full corpus val) |
| Probe-2 (promoted hparams) | ~26M | 1024 | 5k | ~207.2 (at 5k steps; extrapolates to ~60-80 at convergence) |

**Notes:** minigpt-high target was val PPL ≤ 18; actual ~111 indicates underfitting (likely data-loading inefficiencies diagnosed in Phase 4.8). Probe-2 config (dropout 0.0, lr 1e-3, wd 0.05) shows promising convergence trajectory.

## ⚠️ Limitations
- Context length: 1024 tokens (RoPE positional embeddings)
- No instruction tuning or chat finetuning — pure next-token model
- Limited multi-step reasoning; decay quality beyond ~50 tokens
- Token-level frequency of certain patterns may surface sensitive content
- No built-in safety filters — output filtering required at application layer

## 🔒 Safety Considerations
- Output filtered at the application layer (serving middleware)
- No tool use, function calling, or external API access
- Trained on licensed/open datasets with deduplication
- Recommended: add `stop_strings` and `max_new_tokens` guards in deployment
- See `docs/EVAL.md` for full anomaly record and `docs/RUN_NOTES.md` for run-level details

## 📚 Citations
- [TinyStories](https://github.com/ErikSchulz/TinyStories) — Erik Schulz, CC BY-SA 4.0
- [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext) — Salesforce, MIT license
- [FineWeb-Edu](https://huggingface.co/datasets/fineweb) — permissive academic license; 500M token cap applied
- [minigpt_llm](https://github.com/Ankur02Sarkar/minigpt) — Apache-2.0 (this project)

## 📦 Model Weights
- `checkpoints/minigpt-low/best.pt` — 13M params, TinyStories only
- `checkpoints/minigpt-high/best.pt` — 26M params, full corpus
- `checkpoints/minigpt-high/best.pt` from probe-2 run — promoted hparams (dropout 0.0, lr 1e-3, wd 0.05)

---
*Generated from project metadata. For latest results, see docs/EVAL.md and logs/RUN_NOTES.md.*
