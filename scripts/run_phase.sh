#!/usr/bin/env bash
# Run one named training phase (minigpt-low | minigpt-high) with resume.
# Usage: ./scripts/run_phase.sh minigpt-low
set -euo pipefail

NAME="${1:-}"
if [[ "$NAME" != "minigpt-low" && "$NAME" != "minigpt-high" ]]; then
  echo "usage: $0 minigpt-low|minigpt-high" >&2
  exit 2
fi

ROOT="${MINIGPT_ROOT:-/opt/minigpt_llm}"
DATA_DIR="${MINIGPT_DATA_DIR:-/data/tokenized}"
OUT_DIR="${MINIGPT_OUT_DIR:-$ROOT/checkpoints/$NAME}"
LOG_DIR="${MINIGPT_LOG_DIR:-$ROOT/logs/$NAME}"
PYTHON="${MINIGPT_PYTHON:-$ROOT/.venv/bin/python}"
CONFIG="$ROOT/configs/${NAME}.yaml"

mkdir -p "$OUT_DIR" "$LOG_DIR"
START_TS="$(date -Iseconds)"
echo "=== $NAME start $START_TS ===" | tee -a "$LOG_DIR/run.log"

set +e
"$PYTHON" -m training.train \
  --config "$CONFIG" \
  --data-dir "$DATA_DIR" \
  --out-dir "$OUT_DIR" \
  --device cuda \
  --resume latest \
  2>&1 | tee -a "$LOG_DIR/run.log"
EC=${PIPESTATUS[0]}
set -e

END_TS="$(date -Iseconds)"
{
  echo "# RUN — $NAME"
  echo
  echo "- start: $START_TS"
  echo "- end: $END_TS"
  echo "- exit_code: $EC"
  echo "- out_dir: $OUT_DIR"
  echo "- config: $CONFIG"
  if [[ -f "$OUT_DIR/best.pt" ]]; then
    echo "- best.pt: present ($(du -h "$OUT_DIR/best.pt" | awk '{print $1}'))"
  else
    echo "- best.pt: missing"
  fi
  if [[ -f "$OUT_DIR/latest.pt" ]]; then
    echo "- latest.pt: present"
  fi
} | tee "$LOG_DIR/RUN.md"

echo "=== $NAME end $END_TS exit=$EC ===" | tee -a "$LOG_DIR/run.log"
exit "$EC"
