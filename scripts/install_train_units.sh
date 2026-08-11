#!/usr/bin/env bash
# Install systemd unit + heartbeat cron on azure-train (run as ubuntu with sudo).
set -euo pipefail

ROOT="${MINIGPT_ROOT:-/opt/minigpt_llm}"
UNIT_SRC="$ROOT/deploy/systemd/minigpt-train-queue.service"
UNIT_DST="/etc/systemd/system/minigpt-train-queue.service"

if [[ ! -f "$UNIT_SRC" ]]; then
  echo "missing $UNIT_SRC — pull phase/4 branch first" >&2
  exit 1
fi

chmod +x "$ROOT/scripts/"*.sh
sudo cp "$UNIT_SRC" "$UNIT_DST"
sudo systemctl daemon-reload

# Heartbeat every 5 minutes
CRON_LINE="*/5 * * * * MINIGPT_ROOT=$ROOT $ROOT/scripts/heartbeat.sh"
(crontab -l 2>/dev/null | grep -v 'heartbeat.sh' || true; echo "$CRON_LINE") | crontab -

# Initial status
bash "$ROOT/scripts/heartbeat.sh" || true
echo "Installed $UNIT_DST"
echo "Start with: sudo systemctl enable --now minigpt-train-queue"
echo "Logs: journalctl -u minigpt-train-queue -f"
