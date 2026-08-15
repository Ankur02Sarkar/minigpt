#!/usr/bin/env bash
# Phase 4.8/4.4 probe queue: Step B diagnosis + 3 short ~5k-step probes (single GPU).
# Used by systemd minigpt-probe-queue.service — survives SSH disconnect.
set -euo pipefail

ROOT="${MINIGPT_ROOT:-/opt/minigpt_llm}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate" 2>/dev/null || true

export MINIGPT_DATA_DIR="${MINIGPT_DATA_DIR:-/data/tokenized}"
DATA="$MINIGPT_DATA_DIR"
CKPT="$ROOT/checkpoints"

if ! mountpoint -q /data 2>/dev/null; then
  echo "WARNING: /data not a mountpoint — trying mount -a" >&2
  mount -a 2>/dev/null || true
fi
if [[ ! -f "$DATA/fineweb.bin" ]]; then
  echo "FATAL: missing $DATA/fineweb.bin" >&2
  exit 1
fi

# --- Step B: TinyStories val split + baseline diagnosis on the original best.pt ---
echo "=== Step B start $(date -Iseconds) ==="
python -c "
from pathlib import Path
from minigpt_llm.data.paths import DataPaths
from minigpt_llm.data.shards import split_tinystories_val
paths = DataPaths('/data')
if not paths.tinystories_val_bin.is_file():
    split_tinystories_val(paths, train_frac=0.95, force=True)
    print('tinystories_val.bin created')
else:
    print('tinystories_val.bin already exists')
"
mkdir -p "$CKPT"
python -m scripts.diagnose_phase4 \
  --checkpoint "$CKPT/minigpt-high/best.pt" \
  --val-shard "$DATA/val.bin" \
  --context-length 1024 --batch-size 4 --device cuda \
  2>&1 | tee "$CKPT/diagnose_baseline.json"
echo "=== Step B end $(date -Iseconds) ==="

# --- Step C: 3 sequential probes (skip if already complete) ---
run_probe() {
  local name="$1"
  local out="$CKPT/$name"
  mkdir -p "$out"
  if [[ -f "$out/.phase_complete" ]]; then
    echo "skip $name (marker exists)"
    return 0
  fi
  echo "=== $name start $(date -Iseconds) ==="
  python -m training.train \
    --config "configs/$name.yaml" \
    --data-dir "$DATA" \
    --out-dir "$out" \
    --device cuda --resume latest
  date -Iseconds >"$out/.phase_complete"
  echo "=== $name complete $(date -Iseconds) ==="
  # full-val eval on this probe's best.pt (the true PPL, bypassing the 50-batch probe eval)
  python -m scripts.diagnose_phase4 \
    --checkpoint "$out/best.pt" \
    --val-shard "$DATA/val.bin" \
    --context-length 1024 --batch-size 4 --device cuda \
    2>&1 | tee "$out/diagnose.json"
}

run_probe probe-1-sampler
run_probe probe-2-hparam
run_probe probe-3-mix

echo "=== probe queue all done $(date -Iseconds) ==="
exit 0