# Production: one GPU VM

The site and dashboard run on Vercel, and accounts, keys, voices and usage live in Supabase. This VM does the speech work: `api.dhvni.in` → Caddy (HTTPS) → gateway → worker.

| | Where | Listens on |
|---|---|---|
| Caddy | `/etc/caddy/Caddyfile` (copy: `Caddyfile`) | 80, 443. Certificates are automatic |
| Gateway | `vg-gateway.service`, code in `~/voice-gateway/gateway` | 127.0.0.1:8080 |
| Worker | `vg-worker.service`, code in `~/voice-gateway/worker` | 127.0.0.1:8100 |
| Fine-tuned model | `~/models/ritesh-voxcpm2-step250` (base `openbmb/VoxCPM2` + the step 250 LoRA) | |
| Gateway data | `~/voice-gateway/gateway/data`: SQLite and each voice's reference audio. Back it up | |

Configuration:

- `~/voice-gateway/gateway/.env` (mode 600) holds the gateway's settings and keys. Everything is in `gateway/src/voicegw/config.py`.
- `/etc/voice-gateway/worker.env` holds the worker's settings (`worker.env.example`).
- `/etc/voice-gateway/gateway.env` holds the shared `VG_WORKER_TOKEN`.

## First install

These are the steps that set up `acel4` (Ubuntu 22.04, NVIDIA L4, driver 595).

```bash
# on the VM
sudo apt-get install -y ffmpeg && curl -LsSf https://astral.sh/uv/install.sh | sh
# from the Mac: code (git-tracked files) and the private bundle (LoRA + gateway data), over SSH only
git ls-files -co --exclude-standard -z | rsync -a --from0 --files-from=- ./ acel4:voice-gateway/
rsync -a ../voice-gateway-bundle/ acel4:voice-gateway-bundle/ --exclude '*.tar.gz'
# on the VM
cd ~/voice-gateway/worker && uv sync && cd ../gateway && uv sync
cp -R ~/voice-gateway-bundle/gateway-data ~/voice-gateway/gateway/data
cd ~/voice-gateway/worker
BASE=$(uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('openbmb/VoxCPM2'))")
uv run python ../training/merge_lora.py "$BASE" ~/voice-gateway-bundle/lora/ritesh_step250 ~/models/ritesh-voxcpm2-step250
# then the env files above, the two units into /etc/systemd/system, Caddy from its apt repo
sudo systemctl enable --now vg-worker vg-gateway caddy
```

The cloud security group must allow inbound 80 and 443. DNS needs an `A` record `api` → the VM's IP at GoDaddy. Until the record exists, `https://154-201-127-159.sslip.io` serves the same thing.

## Updating

```bash
# from the Mac, in the repo
git ls-files -co --exclude-standard -z | rsync -a --from0 --files-from=- ./ acel4:voice-gateway/
ssh acel4 'cd ~/voice-gateway/gateway && ~/.local/bin/uv sync -q && sudo systemctl restart vg-gateway'
```

Restart `vg-worker` only when worker code or the model changes. It takes about 90 s to load and warm up, and TTS answers 503 meanwhile.

## Checking

```bash
ssh acel4
curl -s localhost:8080/internal/health          # worker model, accounts backend, request log counts
cd ~/voice-gateway/gateway && .venv/bin/voicegw check   # Supabase reachable, live voices have audio here
journalctl -u vg-gateway -f                     # one line per request; usage sync warnings
nvidia-smi                                      # one TTS copy uses about 7 GB
```

## Capacity

- **Measured on the L4,** fine-tuned voice, one TTS copy: first audio in about 80 ms. Rendering runs at 0.85× real time, so one copy keeps one stream fed.
- **Concurrent requests queue for a copy.** Raise `VG_WORKER_TTS_INSTANCES` in `worker.env` when the GPU has about 8 GB free per extra copy.
- The GPU is shared with the `~/voice-cloning` app on port 8000, which uses about 6.8 GB.
