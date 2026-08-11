# Data Licenses

Every dataset used by minigpt-llm is logged here with version, license, and the loader that pulls it.

| Dataset | HF ID / config | License | How we use it | Loader |
|---|---|---|---|---|
| TinyStories | `roneneldan/TinyStories` (train) | CDLA-Sharing-1.0 (see dataset card) | Full download → clean → BPE → `tinystories.bin` | `minigpt_llm.data.download.download_tinystories` |
| WikiText-103 | `wikitext` / `wikitext-103-v1` (train) | CC BY-SA 3.0 | Full download → clean → BPE → `wikitext.bin` + 5% `val.bin` | `minigpt_llm.data.download.download_wikitext` |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` / `sample-10BT` | ODC-By 1.0 | **Stream only**, no raw mirror; filter + tokenize in-flight; **hard cap 500M tokens** → `fineweb.bin` | `minigpt_llm.data.fineweb.stream_and_tokenize_fineweb` |

## Policy

- **Azure-only data plane.** Corpora are never downloaded to developer laptops. All I/O targets the portable managed disk at `/data`.
- **FineWeb is never materialized raw.** Only int32 token IDs are written.
- **Token cap is budget-critical.** Do not raise the FineWeb cap above 500M without revisiting AGENTS.md cost model.
- Update this file in the same PR when adding or removing a dataset.

## Attribution

Please cite the upstream dataset cards when publishing models trained on this mix. See each Hugging Face dataset page for the canonical citation.
