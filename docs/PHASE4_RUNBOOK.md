# Phase 4 Runbook — minigpt-low + minigpt-high

Long training runs on **azure-train** under **systemd**. Your laptop / Grok session can disconnect; training continues.

## Model names

| Name | Config | Steps | Data | Target val PPL |
|---|---|---|---|---|
| **minigpt-low** | `configs/minigpt-low.yaml` | 50k | TinyStories | ≤ 25 |
| **minigpt-high** | `configs/minigpt-high.yaml` | 100k | tiny+wiki+fineweb | ≤ 18 |

Sequential queue: low → high (one GPU).

## Start (Session A)

```bash
# Mac
az vm start -g minigpt-rg -n minigpt-train
ssh azure-train

# On train VM
cd /opt/minigpt_llm
git fetch && git checkout phase/4-training-runs && git pull
source .venv/bin/activate
uv pip install -e .
# mount /data if needed
df -h /data && nvidia-smi
ls /data/tokenized/

bash scripts/install_train_units.sh
sudo systemctl enable --now minigpt-train-queue
systemctl is-active minigpt-train-queue
journalctl -u minigpt-train-queue -n 30 --no-pager
cat /opt/minigpt_llm/STATUS.json
# wait for first train_step / checkpoint, then disconnect
```

**Do not** `az vm deallocate` until both models finish.

## Verify progress (Session B — hours later)

```bash
az vm show --show-details -g minigpt-rg -n minigpt-train --query powerState -o tsv
ssh azure-train 'cat /opt/minigpt_llm/STATUS.json; systemctl is-active minigpt-train-queue; ls -lah /opt/minigpt_llm/checkpoints/minigpt-low /opt/minigpt_llm/checkpoints/minigpt-high 2>/dev/null; journalctl -u minigpt-train-queue -n 40 --no-pager'
```

| Signal | Meaning |
|---|---|
| `service: active` + rising `step` | still training |
| `phase: both_complete` + both `best.pt` | done |
| `service: failed` / `inactive` + missing best | check journal; `sudo systemctl restart minigpt-train-queue` (uses `--resume latest`) |

## After success

```bash
# fill RUN.md already partially written by run_phase.sh
# then free GPU billing:
az vm deallocate -g minigpt-rg -n minigpt-train
```

## Resume after crash

Units use `--resume latest`. Just:

```bash
sudo systemctl restart minigpt-train-queue
```

Queue skips a model if `checkpoints/<name>/.phase_complete` exists.

## Paths

| Artifact | Path |
|---|---|
| Checkpoints | `/opt/minigpt_llm/checkpoints/minigpt-low|high/` |
| Logs | `/opt/minigpt_llm/logs/minigpt-low|high/run.log` |
| Heartbeat | `/opt/minigpt_llm/STATUS.json` |
| Data | `/data/tokenized/` (never on `/mnt`) |
