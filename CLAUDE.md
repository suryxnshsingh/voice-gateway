# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Two Python services plus tooling:

- **`gateway/`** (`voicegw`, CPU) serves two things:
  - The public, branded v1 TTS/STT API (`public_api.py`; customer docs in `docs/API.md`).
  - An internal real-time voice agent over WebSocket (`session.py`, test page at `/playground`).
- **`worker/`** (`vgworker`, GPU) is internal only, called by the gateway. It does VoxCPM2 streaming TTS, emotion2vec tone and ECAPA speaker similarity.
- **`bench/`** holds the STT, LLM and TTS benchmarks. **`training/`** holds the VoxCPM2 LoRA fine-tuning pipeline for teacher voices.

`README.md` is the operator guide: onboarding customers, production settings, measured latencies. `training/README.md` covers fine-tuning.

## Commands

`gateway/`, `worker/` and `bench/` are separate uv projects, each with its own `.venv`. Run commands from inside the project directory.

```bash
scripts/dev.sh                                   # worker on :8100, then gateway on :8000 (enables keep-warm on macOS)

cd gateway
uv run voicegw serve
uv run pytest -q                                 # all gateway tests
uv run pytest tests/test_session.py::test_barge_in_cancels_reply_and_keeps_only_heard_text -q
uv run voicegw create-customer "Name"            # prints the API key once
uv run voicegw import-voice --customer cust_... --name ... --speaker ... --reference f.wav --attestation "..." --operator ...
uv run voicegw replace-reference <voice_id> f.wav --operator ...   # refuses a different speaker
uv run voicegw set-persona <voice_id> --persona-file personas/x.txt --greeting "..."
uv run voicegw talk q1.wav q2.wav --api-key ... --voice ...          # scripted voice-agent turns, prints latencies

cd worker
VG_WORKER_DATA_DIR=./data uv run vgworker        # add VG_WORKER_TTS_MODEL=<merged model dir> to serve a fine-tune
uv run pytest -q
```

- **Smart Turn parity test:** `tests/test_turn.py` needs `transformers`, which only the bench env has. Run it with `cd bench && uv run --with pytest python -m pytest ../gateway/tests/test_turn.py --rootdir ../gateway`.
- **Benchmarks:**
  - `cd bench && uv run python stt_bench.py [--only a,b] [--merge] [--report]`
  - `cd gateway && uv run python ../bench/llm_bench.py --runs 3 [--set tiers]`
  - `cd bench && uv run python tts_sweep.py --voice ... --real ...` (needs the worker running)
- **Training** runs in `training/voxcpm-src/.venv` (the official OpenBMB/VoxCPM repo, gitignored). The sequence is `prepare_dataset.py` → `finalize_dataset.py` → `train_voxcpm_ritesh.py` → `eval_checkpoints.py` → `score_eval.py` → `merge_lora.py`; see `training/README.md`.
- **No linter or formatter** is configured.

## Architecture

### Real-time voice agent (`gateway/src/voicegw/session.py`)

The core is one `Session` per WebSocket: an asyncio state machine. The module docstring documents the client protocol.

- **Input.** Mic PCM (16 kHz) goes to `self.audio` (`PcmRing`, addressed by stream milliseconds) and to the STT stream. The same frames run through `frontend.AudioFrontend` (RNNoise → Silero) in a thread pool. VAD events are `start`, `pause` (200 ms of silence), `resume` and `stop`.
- **STT** (`providers/stt.py` chain: Soniox → OpenAI, configurable).
  - A failing provider is marked down process-wide.
  - Mid-session the session reconnects and **replays the current turn's audio** from `PcmRing`. Token times are shifted by the new stream's `offset_ms`.
  - `OpenAIRealtimeSTT.needs_endpointing` is True: OpenAI has no turn detection, so the session ends turns itself. `finalize()` is a buffer commit.
  - An endpoint that arrives while the user is still speaking is **deferred**, not handled.
- **End of turn.** On a VAD `pause`, Smart Turn (`turn.py`, ONNX) scores the last 8 s. If p ≥ threshold, the session commits now and starts a draft. Otherwise it falls back to `turn_silence_ms`, stretched to 1.2 s when p < 0.15.
- **Responses.**
  - A `Response` owns one task. `_interrupt()` cancels it; the `finally` block in `_response_main` handles history, usage and `response.done`.
  - **Drafts** (`held=True`) start from partial transcripts at a pause. Their events buffer in `held_events` and audio sits in a paused `Playout`.
  - `_end_turn` releases a draft whose words match the final transcript (`_same_words`), otherwise discards it unseen.
  - `pending` means held and not yet confirmed. Only pending drafts may be dropped silently.
- **Output.** LLM deltas go through `text.SentenceSplitter` to worker TTS and into `playout.Playout`. Playout paces audio to real time with about `playout_lead_ms` buffered on the client. That makes replies interruptible until actually heard, and `spoken_text()` exact.
  - Cached **openers** play instantly. They are rendered once per voice, kept in memory and on disk, and keyed by reference file plus worker model.
- **Barge-in has two stages.** VAD speech while the reply is audible pauses playout; the VAD bar is higher during a reply. Real words (`bargein.py` filters backchannels and echo) or long speech cancel the reply. Silence resumes it.
  - **Carry:** if the user kept talking before any audio played, the turn text carries into the next turn.

### Public API (`public_api.py`, `app.py`)

- **Branding is a hard requirement.** Engine names (OpenAI, gpt-*, VoxCPM, Soniox) must never appear in responses, errors, headers, `/v1/usage` or OpenAPI. `tests/test_public_api.py` checks this.
- **Errors:** upstream failures map to neutral messages, in `providers/transcribe.py` for STT and `_engine_error` for TTS.
- **Hidden routes:** internal routes (`/v1/realtime*`, consent, self-serve cloning, playground) use `include_in_schema=False`. Self-serve cloning is gated by `VG_SELF_SERVE_VOICES` (off in v1).
- **STT** is a straight relay of the uploaded file to `VG_STT_RELAY_MODEL`.
- **TTS** splits text with `SentenceSplitter`. It streams pcm/wav directly, and mp3/opus through an ffmpeg pipe (`audio.StreamEncoder`).
- **Limits** (`limits.py`) are in-process per customer. A streaming response holds its slot until the generator finishes.

### Worker (`worker/src/vgworker`)

- **One VoxCPM2 instance renders one sentence at a time** (batch-1 KV cache). `TTSPool._acquire` registers a holder.
- **Release** happens through a done-callback on the render thread's future and is guarded by holder identity.
- **The watchdog** (`app._watchdog`) reclaims instances whose thread finished but was never released, logs long holds, and runs keep-warm.
- **`/tts`** is a WebSocket, one sentence per connection.
- **`PUT /voices/{id}`** stores the reference, builds the prompt cache and renders a warm-up sentence. A voice's first sentence is about 6× slower otherwise.

## Gotchas

- **VoxCPM2 on Apple MPS must run fp32.** bf16 and fp16 glitch, and `torch.compile` is CUDA-only. Expect about 1× real time and 8 GB on a Mac.
- **SpeechBrain switches off TorchScript's profiling executor process-wide,** which breaks VoxCPM's AudioVAE on MPS. `vgworker/speaker.py` sets `SB_DISABLE_QUIRKS=disable_jit_profiling` before importing it; keep that.
- **Macs swap out an idle model under memory pressure,** and the first request then stalls for 10–30 s. Use `VG_WORKER_KEEP_WARM_S`. Don't load a second VoxCPM2 next to a running worker.
- **Training on MPS:** `PYTORCH_MPS_LOW_WATERMARK_RATIO` must be ≤ `PYTORCH_MPS_HIGH_WATERMARK_RATIO` (the low default is 1.4).
- **Soniox tokens are sub-word pieces that carry their own spaces.** Never join them with spaces. OpenAI finals are whole items; the adapter adds a leading space.
- **OpenAI realtime transcription** takes 24 kHz PCM (the adapter resamples) and closes quietly. Adapters must raise `SttError` on any close they didn't request, otherwise failover never fires.
- **Python's `\w` doesn't match Devanagari vowel signs.** Use `bargein._WORD` for word splitting.
- **ffmpeg can't write a valid WAV header to a pipe.** `audio.ffmpeg_convert` decodes to s16le and builds the header in Python.
- **TTS reads Hindi best in Devanagari with English terms in Latin script.** `VOICE_RULES` and the training transcripts follow this. Write numbers as spoken words, not digits.
- **One worker serves one model** (`VG_WORKER_TTS_MODEL`). With a fine-tune loaded, every voice is rendered by that fine-tune.
- **Every voice needs a consent record** (`consents` table). `import-voice` requires `--attestation`, and `replace-reference` refuses a different speaker. Raw recordings and training data are gitignored and must stay out of git.

## Configuration

- **Gateway:** pydantic settings with prefix `VG_` (`gateway/src/voicegw/config.py`). Session-level knobs (VAD, barge-in, drafts, Smart Turn, openers) live in `SessionConfig` in `session.py`.
  - **`.env` is read from the current directory.** That's why the gateway and `bench/llm_bench.py` run from `gateway/`. `bench/stt_bench.py`, `bench/tts_sweep.py` and `training/prepare_dataset.py` load `gateway/.env` explicitly instead.
  - **The local `gateway/.env` (gitignored) overrides several defaults:** OpenAI STT first (Soniox is unfunded), `VG_LLM_SERVICE_TIER=priority` and `VG_PLAYOUT_PREBUFFER_MS=250`. Don't infer runtime behaviour from `config.py` alone. `.env.example` lists the keys.
- **Worker:** env prefix `VG_WORKER_` (`worker/src/vgworker/config.py`). `scripts/dev.sh` starts the **base** model. To serve the fine-tune, export `VG_WORKER_TTS_MODEL=$(pwd)/training/models/ritesh-voxcpm2-step250` first; it's gitignored and exists only locally.

## Runtime data and debugging

- **`gateway/data/`** holds the SQLite DB (customers, hashed keys, voices, consents, usage). Each voice has `voices/<id>/` with its reference, earlier references, consent recordings and the opener cache.
- **`worker/data/voices/`** is the worker's own copy of each reference. The gateway re-pushes a voice when the worker answers `voice_not_found`.
- **Health:**
  - `curl localhost:8000/internal/health` (localhost only) shows engine details.
  - The worker's `/health` lists `busy` holders (who has a TTS instance and for how long).
  - `kill -USR1 <worker pid>` writes all thread stacks to the worker log; py-spy needs root on macOS.
- **Logs:** the gateway logs one line per reply, with status, cancel reason, pauses, `audio_ms` and per-stage timings (`first_text`, `first_tts_chunk`, `first_audio_after_turn_end`, `underruns`).

## Tests

Session tests drive the real orchestration with fakes (`FakeWS`, `FakeLLM`, `FakeWorker`, `make_session` in `tests/test_session.py`), calling `_on_stt` and `_on_vad` directly. Two other files follow the same pattern:

- `tests/test_review_regressions.py` reproduces concurrency bugs found in review. Add one there for any new race.
- Public API tests use FastAPI's `TestClient` and override `app.state` (db, worker, transcriber, limiter) after startup.
