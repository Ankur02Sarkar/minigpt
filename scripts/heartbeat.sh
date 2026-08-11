#!/usr/bin/env bash
# Write /opt/minigpt_llm/STATUS.json for remote health checks (cron every 5 min).
set -euo pipefail

ROOT="${MINIGPT_ROOT:-/opt/minigpt_llm}"
STATUS="$ROOT/STATUS.json"
LOG_LOW="$ROOT/logs/minigpt-low/run.log"
LOG_HIGH="$ROOT/logs/minigpt-high/run.log"

pick_log() {
  if [[ -f "$LOG_HIGH" ]] && grep -q "train_step" "$LOG_HIGH" 2>/dev/null; then
    echo "$LOG_HIGH"
  elif [[ -f "$LOG_LOW" ]]; then
    echo "$LOG_LOW"
  else
    echo ""
  fi
}

LOG="$(pick_log)"
STEP=0
LOSS=null
if [[ -n "$LOG" && -f "$LOG" ]]; then
  STEP="$(grep -oE 'step[= ]+[0-9]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9]+' || echo 0)"
  LOSS="$(grep -oE 'loss[= ]+[0-9.]+' "$LOG" 2>/dev/null | tail -1 | grep -oE '[0-9.]+' || echo null)"
fi

GPU="n/a"
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU="$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || echo n/a)"
fi
DISK_DATA="$(df -h /data 2>/dev/null | awk 'NR==2{print $5}' || echo n/a)"
DISK_ROOT="$(df -h / 2>/dev/null | awk 'NR==2{print $5}' || echo n/a)"
ACTIVE="$(systemctl is-active minigpt-train-queue 2>/dev/null || echo inactive)"
PHASE="unknown"
if [[ -f "$ROOT/checkpoints/minigpt-high/.phase_complete" ]]; then
  PHASE="both_complete"
elif [[ -f "$ROOT/checkpoints/minigpt-low/.phase_complete" ]]; then
  PHASE="high_running_or_pending"
elif systemctl is-active minigpt-train-queue >/dev/null 2>&1; then
  PHASE="low_or_queue_running"
fi

# JSON (loss may be null without quotes)
if [[ "$LOSS" == "null" || -z "$LOSS" ]]; then
  LOSS_JSON=null
else
  LOSS_JSON="$LOSS"
fi

cat >"$STATUS" <<EOF
{"step":${STEP:-0},"loss":${LOSS_JSON},"gpu":"${GPU}","disk_data":"${DISK_DATA}","disk_root":"${DISK_ROOT}","service":"${ACTIVE}","phase":"${PHASE}","ts":"$(date -Iseconds)"}
EOF
