# Phase 1 Runbook (Azure B2ms → T4)

> **Status: Phase 1 complete (2026-08-11).** Shards on `/data/tokenized/`, prep deallocated, train VM has T4 + data disk. **Deallocate `minigpt-train` when not training.**

Code is developed on the Mac. **All dataset I/O ran on `minigpt-prep` (B2ms)** with `/data` mounted; disk then moved to `minigpt-train`.

## Prerequisites

- SSH: `Host azure-prep` → prep VM public IP
- Repo on VM: `/opt/minigpt_llm`
- `HF_TOKEN` set on the VM (FineWeb / gated rate limits)
- Branch: `phase/1-data-pipeline` (or `main` after merge)

## Install / update code

```bash
# from Mac
git push -u origin phase/1-data-pipeline

ssh azure-prep
cd /opt/minigpt_llm
git fetch && git checkout phase/1-data-pipeline && git pull
source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
```

## Run the pipeline

Implementation order trains BPE **before** FineWeb tokenization (AGENTS task IDs 1.4/1.6 are reordered in code).

```bash
# long-running: use tmux
tmux new -s phase1
export HF_TOKEN=...   # from Key Vault / local secret — never commit
python -m scripts.run_phase1_pipeline --data-root /data
```

Useful flags:

| Flag | Purpose |
|---|---|
| `--force` | Overwrite existing artifacts |
| `--skip-fineweb` | Dry run without the multi-hour stream |
| `--fineweb-token-cap N` | Override 500M cap (tests / partial runs) |
| `--only clean train_bpe …` | Run a subset of steps |

Resume: steps whose outputs already exist are skipped unless `--force`.

## Smoke test only

```bash
python -m training.dataset \
  --shard /data/tokenized/tinystories.bin \
  --context 1024 \
  --batch 4
```

## Verify exit criteria

```bash
ls -lah /data/tokenized/
# tinystories.bin wikitext.bin fineweb.bin val.bin meta.pkl
df -h /data
# aim ≤ 20 GB used; delete /data/raw after successful tokenize if tight
python -c "import pickle; print(pickle.load(open('/data/tokenized/meta.pkl','rb')))"
```

## Optional disk cleanup

After shards + meta look good:

```bash
# free ~8 GB raw if needed for the ≤20 GB gate
sudo rm -f /data/raw/tinystories.txt /data/raw/wikitext103.txt
```

Never store artifacts under `/mnt` (wiped on deallocate).

## 1.10 — Move data disk to T4 (when ready for GPU)

**Billing warning:** creating/starting `Standard_NC8as_T4_v3` is ~$0.75/hr. Only do this when data is verified and you are ready for Phase 2+.

```bash
# from Mac
az vm disk detach -g minigpt-rg -n minigpt-prep --name minigpt-data
az vm deallocate -g minigpt-rg -n minigpt-prep

az vm create -g minigpt-rg -n minigpt-train \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size Standard_NC8as_T4_v3 \
  --admin-username ubuntu \
  --ssh-key-values ~/.ssh/id_ed25519.pub \
  --public-ip-sku Standard \
  --os-disk-size-gb 128 \
  --os-disk-sku StandardSSD_LRS

az vm extension set -g minigpt-rg -n minigpt-train \
  --name NvidiaGpuDriverLinux --publisher Microsoft.HpcCompute

az vm disk attach -g minigpt-rg -n minigpt-train --name minigpt-data
```

On the train VM: format is already done — mount `/data` via fstab (same UUID/device as prep; check `lsblk`), verify:

```bash
df -h /data
nvidia-smi   # 1× Tesla T4 16 GB
cp /data/vocab/* /opt/minigpt_llm/tokenizer/   # after repo clone
```

**Deallocate the T4 when idle.** `az vm deallocate -g minigpt-rg -n minigpt-train`
