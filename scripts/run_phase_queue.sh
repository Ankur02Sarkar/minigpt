#!/usr/bin/env bash
# Sequential train: minigpt-low then minigpt-high (single GPU).
# Used by systemd minigpt-train-queue.service — survives SSH disconnect.
set -euo pipefail

ROOT="${MINIGPT_ROOT:-/opt/minigpt_llm}"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/.venv/bin/activate" 2>/dev/null || true

export MINIGPT_ROOT="$ROOT"
export MINIGPT_DATA_DIR="${MINIGPT_DATA_DIR:-/data/tokenized}"
export MINIGPT_PYTHON="${MINIGPT_PYTHON:-$ROOT/.venv/bin/python}"

MARKER_LOW="$ROOT/checkpoints/minigpt-low/.phase_complete"
MARKER_HIGH="$ROOT/checkpoints/minigpt-high/.phase_complete"

run_one() {
  local name="$1"
  local marker="$2"
  if [[ -f "$marker" ]]; then
    echo "skip $name (marker $marker exists)"
    return 0
  fi
  echo "queue: starting $name at $(date -Iseconds)"
  bash "$ROOT/scripts/run_phase.sh" "$name"
  local ec=$?
  if [[ $ec -eq 0 ]]; then
    mkdir -p "$(dirname "$marker")"
    date -Iseconds >"$marker"
    echo "queue: $name complete"
  else
    echo "queue: $name failed exit=$ec" >&2
    return "$ec"
  fi
}

# Ensure data is mounted
if ! mountpoint -q /data 2>/dev/null; then
  echo "WARNING: /data not a mountpoint — trying mount -a" >&2
  mount -a 2>/dev/null || true
fi
if [[ ! -f /data/tokenized/tinystories.bin ]]; then
  echo "FATAL: missing /data/tokenized/tinystories.bin" >&2
  exit 1
fi

run_one minigpt-low "$MARKER_LOW"
run_one minigpt-high "$MARKER_HIGH"
echo "queue: all phases done at $(date -Iseconds)"
exit 0
