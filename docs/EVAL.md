# Phase 4 — Final Evaluation Report

**Date:** 2026-08-15 · **Branch:** `phase/4-training-runs` · **Hardware:** 1× Tesla T4 16 GB (`minigpt-train`, `Standard_NC8as_T4_v3`) · **Precision:** FP16 autocast (no BF16 on Turing)

**TL;DR:** Both training runs completed cleanly (exit 0) under the systemd queue, and both `best.pt` checkpoints exist. However, **both runs missed their planned val-PPL targets** — the minigpt-low target was structurally invalid (train/val domain mismatch), and minigpt-high underfits the corpus (PPL ≈ 111 vs ≤ 18). Generation samples corroborate the metrics. The T4 VM was **deallocated 2026-08-15 ~13:37 UTC** after ~11 h idle. Diagnosis is tracked as AGENTS.md task **4.8**.

---

## Runs at a glance

| | **minigpt-low** | **minigpt-high** |
|---|---|---|
| Config | `configs/minigpt-low.yaml` | `configs/minigpt-high.yaml` |
| Architecture | 6L · 256-dim · 4 heads · ctx 512 | 8L · 384-dim · 6 heads · ctx 1024 |
| Params | 13,012,224 | 26,450,304 |
| Train data | TinyStories (444.7M tokens) | TinyStories + WikiText + FineWeb-Edu (1.06B tokens) |
| Steps | 50,000 | 100,000 |
| Tokens/step | 64 × 512 = 32,768 | 64 × 1024 = 65,536 |
| Tokens seen | 1.64B (≈ 3.7 epochs) | 6.55B (≈ 6.2 epochs) |
| Wall time | 9 h 24 m (2026-08-11 10:14 → 19:38 UTC) | 78 h 38 m (2026-08-11 19:38 → 2026-08-15 02:16 UTC) |
| Throughput | ~50.9k tok/s | ~23.2k tok/s |
| Peak LR / min | 5e-4 → 5e-5 (cosine, 2k warmup) | 5e-4 → 5e-5 (cosine, 2k warmup) |
| Final train loss | — (see logs) | ~4.2 |
| **Best val loss** | **9.5987** (step 1,000) | **4.7136** (step 100,000) |
| **Best val PPL** | **~14,745** | **~111.4** |
| Target val PPL | ≤ 25 — **invalid target** (see below) | ≤ 18 — **missed** |
| Exit code | 0 | 0 |
| Checkpoint | `checkpoints/minigpt-low/best.pt` (149 MB) | `checkpoints/minigpt-high/best.pt` (303 MB) |

Val set for both: `val.bin` = WikiText-103 5% split (5.75M tokens), sequential windows (102.4k–204.8k tokens scored per eval).

## Val-loss trajectories

**minigpt-low** (eval every 1k steps; sampled):

| step | val_loss | val_ppl |
|---|---|---|
| 1,000 | 9.599 | 14,745 |
| 11,000 | 10.151 | 25,610 |
| 21,000 | 10.306 | 29,921 |
| 31,000 | 10.018 | 22,436 |
| 41,000 | 9.989 | 21,795 |
| 50,000 | 9.726 | 16,744 |

Val loss **rose** through mid-training as the model specialized on TinyStories, then partially recovered. `best.pt` is therefore the **step-1,000** checkpoint.

**minigpt-high** (eval every 2k steps; sampled):

| step | val_loss | val_ppl |
|---|---|---|
| 2,000 | 5.880 | 358.0 |
| 22,000 | 4.941 | 139.9 |
| 42,000 | 4.867 | 129.9 |
| 62,000 | 4.805 | 122.1 |
| 82,000 | 4.735 | 113.9 |
| 100,000 | 4.714 | 111.4 |

Monotonically decreasing and **still improving at the final step**; `best.pt` = final state.

## Target analysis

### minigpt-low — target invalid, not merely missed

The plan (AGENTS.md 4.2) set "target val PPL ≤ 25", but the config trains on TinyStories only while the only val shard is WikiText. That is a cross-domain eval: near-uniform loss on the val set is expected (ln 32,000 ≈ 10.37; observed 9.6–10.3). No amount of TinyStories training can hit ≤ 25 on WikiText. **Action:** future low-model evals need a held-out TinyStories val split (see task 4.8).

### minigpt-high — genuine underfitting

- Train/val gap at the end is small (**4.2 vs 4.71**) → not overfitting despite 6.2 epochs.
- Val loss still falling at step 100k → the run was not "converged at the budget limit" in a healthy sense; the optimization/data pipeline is leaving large gains on the table.
- A 26M-param model exposed to 6.55B tokens should reach WikiText PPL in the 20s–30s even with a suboptimal mix; ~111 with incoherent generations (see `docs/SAMPLES.md`) points at a **pipeline issue**.

**Suspects (to verify in task 4.8):**

1. **Shard ordering / shuffling** — `MultiShardDataset` iterates shards sequentially per epoch; WikiText is ~10% of the mix while the val set is 100% WikiText.
2. **BPE vocab quality** — byte-fallback artifacts are visible in generations (`ï¿½`); audit merge stats and token fertility.
3. **Eval harness** — sanity-check `training/evaluate.py` sequential-window loss against a manual pass on a few batches.
4. **Regularization/optimization** — dropout 0.1 + weight-decay 0.1 over 6+ epochs of a small corpus; LR 5e-4 at effective batch 64.

## Generation quality

See [`docs/SAMPLES.md`](SAMPLES.md). Summary: both models emit topical vocabulary in broken word order and collapse into repetition loops under greedy decoding — consistent with the PPL numbers. Repetition-loop detection (Phase 5.3) would flag every greedy sample.

## Cost record

| Item | Planned (§8.2) | Actual |
|---|---|---|
| minigpt-low T4 time | ~4 h | 9.4 h |
| minigpt-high T4 time | ~20 h | 78.6 h |
| Idle before deallocate | — | ~11.3 h |
| Sample generation (CPU-bound, VM up) | — | ~0.5 h |
| **Total T4 compute** | **~24 h ≈ $18** | **≈ 100 h ≈ $75** |

Root cause of the overrun: sustained throughput was ~23.2k tok/s on the high run while the plan implicitly assumed roughly double, and 100k steps × 65,536 tok/step = 6.55B tokens. Compute spend lands **above the $35–50 project target but inside the $200 free-trial credit**; phases 5–7 should favor CPU/short-burst T4 usage. Verify the exact figure with `az consumption usage list` once billing settles.

## Artifacts

| Artifact | Location |
|---|---|
| Checkpoints (persist on OS disk, VM deallocated) | `/opt/minigpt_llm/checkpoints/minigpt-{low,high}/` |
| Run logs | `/opt/minigpt_llm/logs/minigpt-{low,high}/run.log` + `RUN.md` |
| TensorBoard events | `/opt/minigpt_llm/checkpoints/minigpt-{low,high}/tb/` |
| Heartbeat | `/opt/minigpt_llm/STATUS.json` (`phase: both_complete`) |
| Sample generations (verbatim) | [`docs/SAMPLES.md`](SAMPLES.md) · raw: `logs/phase4-artifacts/` (gitignored) |
| Sample tool | `scripts/generate_samples.py` |

## Next steps

1. **Task 4.8** (new): PPL diagnosis — eval-harness sanity check, dataset shuffle/order audit, BPE audit, TinyStories val split; budget-gated short probes only.
2. **Task 4.4** (optional): short 5k-step tuning probes if 4.8 identifies a fixable cause and credit allows.
3. Phase 5 (inference) can proceed against `checkpoints/minigpt-high/best.pt` as-is — quality caveots documented here.
