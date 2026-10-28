# voice-gateway

A real-time voice endpoint: one WebSocket carries speech-to-text, an LLM reply, text-to-speech in a cloned voice, and tone detection. Users can interrupt it, and it ignores coughs, "haan"s and its own echo.

```
mic ─▶ RNNoise ─▶ Silero VAD ─▶ barge-in controller ─▶ playout buffer ─▶ client speaker
  └─ raw PCM ─▶ STT (Soniox ⇄ OpenAI) ─▶ words, end of turn ┘   ▲
                     │                                       │
                     ├─▶ emotion2vec (acoustic tone) ─┐      │
                     │                                ├─▶ LLM ─▶ sentences ─▶ VoxCPM2 (cloned voice)
                     └─▶ transcript ──────────────────┘
```

There are two services:

| | Runs on | Does |
|---|---|---|
| `gateway/` (`voicegw`) | CPU | The realtime session, STT with failover (Soniox, OpenAI), the LLM, noise suppression, VAD, barge-in, playout, the voices and consent API, API keys, usage metering, and the [playground](http://127.0.0.1:8000/playground) |
| `worker/` (`vgworker`) | GPU | VoxCPM2 streaming TTS, emotion2vec+ tone, and ECAPA speaker similarity for the consent check. It's internal: only the gateway calls it |

## v1 public API (TTS + STT)

The customer-facing API guide is [`docs/API.md`](docs/API.md), and an interactive reference is served at `/docs`.

- **What customers see:** the brand "Dhvni" and the models `dhvni-voice-1` and `dhvni-transcribe-1`. Engine names never appear in responses, errors, headers, `/v1/usage` or the generated docs; tests check for leaks. Set the names with `VG_BRAND`, `VG_BRAND_TTS_MODEL` and `VG_BRAND_STT_MODEL`.
- **Endpoints:**
  - `POST /v1/tts` returns a file or a stream (mp3, opus, wav or pcm).
  - `WS /v1/tts/stream` takes text as it's written and returns PCM.
  - `POST /v1/stt` relays the file unchanged to `VG_STT_RELAY_MODEL` (default `gpt-transcribe`) and returns the text under the brand.
  - `GET /v1/voices`, `/v1/models` and `/v1/usage`.
- **Measured:** first audio in about 80 ms on an NVIDIA L4 (fine-tuned voice, warm), rendering at 0.85× real time with one TTS copy. On the M5 Pro: 300–480 ms.
- **Limits per customer:** 120 requests a minute, 4 concurrent TTS, 8 concurrent STT, 5,000 characters and 25 MB. They're in-process, so move them to Redis once there's more than one gateway.

### Accounts: the Dhvni dashboard (Supabase)

With `VG_SUPABASE_URL` and `VG_SUPABASE_SECRET_KEY` set, the public API takes its accounts from the dashboard's Supabase project (`~/Developer/1cr/dhvni-web`, whose README has the contract). Without them, it uses the gateway's own SQLite (development and tests).

- **API keys** are the ones customers create at `/dashboard/keys`. The gateway checks them with `verify_api_key()`. An unknown or revoked key gets 401; an account that is pending or suspended gets 403.
- **Voices** come from the `voices` table: an account's own, plus shared ones (no `user_id`). The audio stays on the gateway, in `gateway/data/voices/<id>/reference.wav`. A voice that is live in Supabase but has no audio here gets 503; `voicegw check` lists them.
- **Usage.** Every TTS, streaming TTS and STT request that passes authentication is written to the local `requests` table first, failures included. `accounts.UsageSync` then copies rows to `usage_events` every 2 s, in batches, with `on conflict (request_id) do nothing`. If Supabase is down, rows wait and go out later. The database prices them; failed requests cost nothing.
- **Caching.** Answers are cached for `VG_ACCOUNTS_CACHE_S` (30 s), so a revoked key or a new voice takes up to that long. While Supabase is unreachable, the last answer is used for up to 10 minutes.
- **Credit.** `VG_REQUIRE_CREDIT=true` refuses TTS and STT with 402 once an account's prepaid balance (`credit_balance()`) is used up. It is off by default.
- The internal voice agent (`/v1/realtime`) and self-serve cloning still use the local database.

**Onboarding a customer** (operators):

1. They sign up on the site; an admin approves them at `/admin/accounts`; they create their own keys.
2. For a voice, they send a voice request from `/dashboard/voices` with the recordings and consent. On the gateway machine, import it. `--customer` is any local customer (the dashboard decides who may use it):
   ```bash
   cd gateway
   uv run voicegw import-voice --customer cust_... --name "Priya Ma'am" --speaker "Priya Sharma" \
       --reference priya_20s.wav --attestation "Voice request #12, consent checked 2026-10-28" --operator "<you>"
   uv run voicegw replace-reference <voice_id> better.wav --operator "<you>"   # better recording, same person only
   ```
3. Paste the printed voice id into "Make it live" at `/admin/voices`. It shows up in their `GET /v1/voices` within 30 s.

Without Supabase, `uv run voicegw create-customer "Acme Coaching"` prints a local customer and key. Self-serve cloning (`POST /v1/voices` and the consent recording) is off in v1; set `VG_SELF_SERVE_VOICES=true` to switch it on.

**Production settings** (the running setup is in [`deploy/`](deploy/README.md)):
- `VG_SUPABASE_URL` and `VG_SUPABASE_SECRET_KEY` connect the dashboard's accounts.
- `VG_PLAYGROUND_ENABLED=false` hides the test page and the voice-agent WebSocket.
- `VG_SECRET_KEY` must be a fixed random value.
- `VG_WORKER_TOKEN` must be set on both services.
- Run the worker on a GPU (bf16) with `VG_WORKER_KEEP_WARM_S=0`.
- The privacy policy and data-processing agreement must list the speech-to-text engine provider as a sub-processor.

## Run it locally

You need `uv` and `ffmpeg`. On first start the worker downloads about 6 GB of model weights.

```bash
cp gateway/.env.example gateway/.env      # Soniox key, LLM key, a random VG_SECRET_KEY
scripts/dev.sh                            # worker on :8100, gateway on :8000

cd gateway
uv run voicegw create-customer "Arivihan"  # prints a customer id and an API key
open http://127.0.0.1:8000/playground      # paste the key, clone a voice, press Connect, talk
```

The playground gives you a mic session with live transcript, tone chips and latency per reply. It has a box to make the voice say any text, and a clone-a-voice panel that runs the consent flow. Use headphones so you can talk over the reply. If you're on speakers, tick "Mute mic while it speaks".

`uv run voicegw talk q1.wav q2.wav --api-key … --voice …` plays audio files in as user turns and saves the spoken replies. It's useful for scripted latency checks.

## Realtime protocol

Connect to `wss://…/v1/realtime`:
- **Servers** send `Authorization: Bearer <api key>`.
- **Browsers** call `POST /v1/realtime/sessions` with the key, then connect with `?token=…`. The token lasts 2 minutes.

| Direction | Message | Meaning |
|---|---|---|
| → | binary | PCM16 LE mono 16 kHz mic audio, 20–100 ms frames |
| → | `session.update {session}` | voice, persona instructions, LLM on/off, tone, barge-in tuning (fields below) |
| → | `response.create {text?}` | speak `text` (bring your own LLM), or have the LLM reply |
| → | `response.cancel`, `input_audio.commit` | stop the reply; end the user's turn now (push-to-talk) |
| ← | `transcript.delta` / `transcript.done` | live and final user text; backchannels and echo come with `ignored` set |
| ← | `tone` | `source: acoustic` (emotion2vec, fast), then `source: fused` (audio + words, LLM-labelled) |
| ← | `response.started` / `.text.delta` / `.audio.started` / binary / `.done` | the reply. Audio is PCM16 at `output_sample_rate`, sent at playback speed |
| ← | `output_audio.paused` / `.resumed` | barge-in stage 1: hold playback, or carry on |
| ← | `input.speech_started` / `.speech_stopped` | VAD state, for UI |

The full event list is in the docstring at the top of `gateway/src/voicegw/session.py`.

Session fields, all optional:

| Field | Default | |
|---|---|---|
| `voice_id` | none | an active voice on the account |
| `instructions` | "" | persona, appended to the built-in voice rules |
| `llm` / `model` | true / `VG_LLM_MODEL` | `llm: false` means you send `response.create {text}` yourself |
| `reasoning_effort` | `VG_LLM_REASONING_EFFORT` | `"none"` for reasoning models (gpt-5.x/6.x). gpt-6-luna's default reasoning adds ~1 s before the first word |
| `turn_silence_ms` | 800 | silence that ends a turn when the STT has no turn detection of its own (OpenAI) |
| `preemptive` | true | draft the reply from the live transcript as soon as the user pauses; release it when the turn is confirmed, discard it unseen otherwise |
| `tone`, `tone_fusion` | true, true | |
| `output_sample_rate` | 24000 | 16000, 24000 or 48000 |
| `language_hints` | `["en","hi"]` | applies before the first audio frame |
| `noise_suppression`, `denoise_stt` | true, false | RNNoise feeds the VAD. STT gets raw audio unless `denoise_stt`, because denoising artifacts can raise word error |
| `vad`, `vad_threshold`, `vad_start_ms`, `vad_stop_ms` | true, 0.6, 200, 500 | |
| `interrupt_min_words` | 1 | real (non-backchannel) words that confirm an interruption |
| `interrupt_confirm_ms` | 800 | continuous speech that confirms one before any words arrive. 0 means words only |
| `false_interruption_ms` | 700 | after speech stops, resume if nothing real was said |
| `backchannels`, `echo_guard` | true, true | |

## Speech-to-text providers

`VG_STT_PROVIDERS` sets the preference order (default `["soniox","openai"]`):

- **Failover.** A provider that fails with a bad key, no credit or rate limiting (401/402/429) is skipped for 10 minutes, process-wide. The switch also happens mid-session. The current turn's audio is replayed from the gateway's buffer into the next provider, so no words are lost, and the client gets `stt.connected {provider}`.
- **Soniox** detects the end of a turn itself.
- **OpenAI** (`gpt-live-transcribe` by default, set with `VG_OPENAI_STT_MODEL`) streams words while the user talks but has no turn detection, so the gateway ends turns itself (see Latency).
  - It runs with `delay: "minimal"` and `VG_STT_KEYWORDS`. Measured on the 10-clip benchmark: first words arrive 357 ms after speech starts instead of 1055 ms, and accuracy improved too (skeleton error 0.072 vs 0.116).
  - The final text arrives about 0.7 s after a commit.
- **Benchmarks.** See `bench/` below for how the candidates compare.

## Barge-in

1. **Pause.** Silero (fed by RNNoise) hears 200 ms of speech while the assistant is audible. The playout buffer stops at once and the client holds what it has queued.
2. **Confirm or resume:**
   - **Cancel** if STT returns a real word, or the speech runs past 800 ms. The cancelled reply is kept in history only up to what the user actually heard, so the LLM doesn't assume they heard the rest.
   - **Resume** if the speech stops and nothing real was said within 700 ms (a cough, a door). Playback continues from the exact sample it paused on.
   - **Resume, no turn** for a backchannel ("haan", "okay", "हाँ जी"). It's dropped and never becomes a turn.
   - **Ignore** if the "user" is saying the assistant's own words (speakers, no echo cancellation).
3. **Merge.** If the user keeps talking before any reply audio has played, the two fragments become one turn.

The server-side playout buffer is what makes this work. TTS renders faster than real time, but audio goes out at playback speed with about 200 ms queued on the client. So a reply stays interruptible until it has actually been played, and "what was heard" is known to the frame.

## Latency

- **End of turn.** Smart Turn v3.2 (Pipecat, BSD-2, 8 MB ONNX, about 27 ms on CPU) listens to the last 8 s at a 200 ms pause. If it's ≥ 0.5 sure the speaker is finished, the turn ends at once. Otherwise the turn waits for `turn_silence_ms` (800 ms), or 1.2 s if it's < 0.15 sure (they sound mid-thought). Speaking again after a pause discards any draft. On Ritesh Sir's audio, mid-sentence cuts scored a median of 0.01, so a high score is trustworthy. The log-mel features are reimplemented in numpy and match transformers' WhisperFeatureExtractor to 2e-5.
- **Openers** (`openers`). Eight short openers ("अच्छा,", "देखो बेटा,", ...) are rendered once per voice. When a reply begins with one, its audio plays immediately and the rest renders behind it.
- **LLM.** Idle connections are kept for 5 minutes instead of the SDK's 5 s, and a connection is opened as soon as the user starts talking. `VG_LLM_SERVICE_TIER=priority` costs 2× the LLM price and reached the first sentence 205 ms sooner (928 vs 1133 ms median, `bench/llm_bench.py --set tiers`).
- **Early drafts** (`preemptive`). The reply is drafted from the live transcript the moment the VAD hears a pause, and held until the turn is confirmed. LLM and first-sentence TTS time then overlap with end-of-turn detection. A draft is discarded unseen if the user keeps talking or the final words differ. When late STT words arrive, the draft is redone; each redo costs one extra partial LLM call.
- **Tone** never delays the reply (`VG_TONE_WAIT_MS=0`). It is attached to the turn for later replies.
- **Measured** end of speech → first audio, M5 Pro, OpenAI STT, gpt-5.4-mini:
  - 5.1–5.7 s at the start (no drafts).
  - 3.0–3.5 s with drafts.
  - **1.9–2.3 s** with low-delay STT, Smart Turn, openers, warm connections, the priority tier and the fine-tuned voice. That last figure includes the first turn of a fresh session, and no reply stuttered.
  - Most of what remains is waiting for the final transcript (~0.7–1 s) and the LLM's first token (~0.8 s).

## Voice quality

- **Hindi text.** The LLM is told to write Hindi in Devanagari, with English terms in Latin script. VoxCPM2 reads Romanized Hindi with an English accent.
- **Chunking.** TTS chunks are whole sentences, merged up to ~80 characters. The first chunk is cut at a clause only if it is long. Every cut restarts the voice's intonation.
- **Diffusion steps.** The worker default is 8 (`VG_WORKER_TTS_TIMESTEPS`). Measured with `bench/tts_sweep.py`: 12% faster than 10 with the same similarity (0.816 vs 0.810); 4 drops it to 0.729. Requests may override `inference_timesteps` and `cfg_value`.
- **Sample rate.** VoxCPM2 is 48 kHz. Ask for `output_sample_rate: 48000` where bandwidth allows; the playground does.
- **Precision on Macs.** Don't run VoxCPM2 in half precision on Apple GPUs: VoxCPM itself warns that bf16/fp16 drift on MPS produces glitched audio. On CUDA, bf16 is fine.
- **Swapping a reference.** `voicegw replace-reference <voice> <file> --start --end --operator` swaps in a better recording of the same person. It is refused if the speaker doesn't match, and the old reference is kept for rollback.

## Personas

`voicegw set-persona <voice> --persona-file personas/ritesh_sir.txt --greeting "नमस्ते बेटा! ..."` stores default instructions and a greeting with a voice. Sessions that send no `instructions` get the persona. `session.update {"greet": true}` makes the voice speak first; the playground does this.

## Fine-tuned voices

See `training/README.md`. It covers dataset building from long recordings, LoRA fine-tuning VoxCPM2 on the Mac (official trainer, MPS fp32, about 11 s/step), checkpoint scoring, and merging for the worker.

## Voice cloning and consent

A clone can't be used until the person whose voice it is consents in that same voice:

1. `POST /v1/voices`: upload `reference` (10–60 s), plus `name`, `speaker_name` and `language` (`en`/`hi`). This returns a consent phrase with three random code words.
2. `POST /v1/voices/{id}/consent`: upload a `recording` of the speaker reading the phrase. It passes only if Soniox hears every code word (so the recording is fresh, not replayed) and ECAPA says it's the same speaker as the reference (cosine ≥ 0.40). You get 5 attempts within 24 h.
3. `DELETE /v1/voices/{id}`: revokes the voice and deletes the reference audio. Consent records and recordings are kept for audit.

When consent was collected outside the API, for example a teacher's signed release, an operator records that explicitly:

```bash
uv run voicegw import-voice --customer cust_… --name "Ritesh Sir" --speaker "Ritesh …" \
  --reference ritesh.wav --attestation "Release form, HR drive, signed 2026-10-20" --operator "Suryansh"
```

`--shared` makes a voice usable by every customer. `uv run voicegw consents <voice_id>` shows the audit trail.

## Usage metering

- **Public API:** one row per request in `requests` (endpoint, characters, audio seconds, status, time to first audio), copied to the dashboard's `usage_events` when Supabase is set. `GET /v1/usage?since=&until=` returns the billable totals: `tts_characters`, `tts_seconds` and `stt_seconds` of successful requests.
- **Voice agent:** every session records `stt_seconds`, `llm_input_tokens`, `llm_output_tokens`, `tts_characters`, `tts_seconds` and `tone_calls` per customer, with provider and model, in `usage_events`.
- `voicegw usage <customer>` shows both. Billing is meant to be one rate card, the same for every customer, including Arivihan.

## Measured (M5 Pro, VoxCPM2 on MPS fp32, no CUDA)

| | |
|---|---|
| TTS first chunk, warm voice | 190–260 ms |
| TTS first chunk, a voice's first sentence | 1.3 s (now paid when the voice loads, not in a reply) |
| TTS speed | 0.85× real time (just keeps up). A CUDA GPU with bf16 and torch.compile should be several times faster (not measured yet) |
| LLM first token (OpenAI, from India) | 1.0–1.3 s |
| Silero VAD | 2.5 ms CPU per second of audio |
| RNNoise + resampling | about 40 ms CPU per second of audio (≈4% of a core per session) |
| VAD speech-start latency | ~240 ms after speech begins (200 ms `vad_start_ms` + framing) |
| Noise floor with RNNoise | −29 dBFS → −75 dBFS on hiss + hum |

## Benchmarks (`bench/`)

```bash
cd bench
uv run python stt_bench.py      # STT: 8 exact-script clips (en / hi / hinglish) + 2 real teacher clips
cd ../gateway && uv run python ../bench/llm_bench.py --runs 3   # reply LLMs on the real voice prompt
```

- **STT.** Streaming models get the audio at real-time pace, with turns cut by the gateway's own VAD logic. Batch and local models get whole clips. Accuracy is the voice-clone-poc's script-agnostic consonant-skeleton error. Results go to `bench/results/`.
- **LLM.** The benchmark measures time to the first speakable sentence (when TTS can start) and adherence to the voice rules.

## Tests

```bash
cd gateway && uv run pytest     # session flow, drafts, smart turn, openers, barge-in, STT failover, playout, VAD, public API, Supabase accounts
cd worker && uv run pytest      # TTS pool: instances always come back, leaks heal
```

The session tests run the real orchestration with fake STT, LLM and worker. Covered: tone → reply → audio, pause → words → cancel, cough → resume, long speech → cancel without words, backchannel → resume with no turn, echo ignored, a reply staying interruptible until played, and turn merging.

## Before production

- **Provider accounts.** Use the subsidiary's own Soniox and OpenAI accounts rather than Arivihan's, so its costs are its own.
- **GPU sizing.** One VoxCPM2 instance renders one sentence at a time (its KV cache has batch size 1), at roughly 5 GB of VRAM in bf16. Size `VG_WORKER_TTS_INSTANCES` to the number of sentences rendering at once, not to the number of open sessions.
- **Storage.** SQLite and local files work for one gateway box. Move to Postgres and S3 for more than one. Set `VG_WORKER_TOKEN` on both services.
- **Turn detection.** End of turn comes from Soniox endpointing. A semantic turn-detection model (like LiveKit's) would cut mid-sentence pauses being treated as turn ends.
- **Hindi script.** VoxCPM2's handling of Devanagari vs Roman Hinglish should be checked with the voice-clone-poc samples before choosing which script the LLM writes in.
- **Consent hardening.** The consent check can't yet tell a live human from a synthetic voice reading the phrase. Synthetic-speech detection on the consent recording is the next step.
