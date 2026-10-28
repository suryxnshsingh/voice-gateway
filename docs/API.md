# Dhvni API — v1

Text-to-speech and speech-to-text for Hindi, English and Hinglish.

- **Base URL:** `https://api.dhvni.in`. For local testing it's `http://127.0.0.1:8000`.
- **Interactive reference:** `/docs`.
- **Models:** `dhvni-voice-1` (text-to-speech) and `dhvni-transcribe-1` (speech-to-text).

## Authentication

Every request needs an API key from the dashboard (`/dashboard/keys`):

```
Authorization: Bearer vg_...
```

Browsers can't set headers on WebSockets. In a browser, call `POST /v1/tokens` from your backend, then connect with `?token=...`. A token is valid for 2 minutes.

## Voices

Your voices are set up for you by our team. List them to fill a voice picker:

```bash
curl https://api.dhvni.in/v1/voices -H "Authorization: Bearer $KEY"
```
```json
{"voices": [{"id": "voice_19d9ec2eae586ac7", "name": "Ritesh Sir", "status": "active", "created_at": 1792966000}]}
```

## Text-to-speech

### `POST /v1/tts`

| Field | Default | |
|---|---|---|
| `voice_id` | required | from `/v1/voices` |
| `text` | required | up to 5,000 characters. Hindi reads best in Devanagari, with English words in English letters: `"देखो बेटा, mitochondria cell का powerhouse है।"` Write numbers the way they should be spoken |
| `format` | `mp3` | `mp3`, `opus` (Ogg), `wav`, or `pcm` (raw 16-bit little-endian mono) |
| `sample_rate` | `24000` | `16000`, `22050`, `24000`, `44100` or `48000` |
| `quality` | `fast` | `fast` or `hq` (slightly higher quality, a little slower) |
| `stream` | `false` | `true`: audio starts arriving in about 0.3 s and keeps coming as it's generated |

```bash
curl https://api.dhvni.in/v1/tts -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"voice_id":"voice_19d9ec2eae586ac7","text":"नमस्ते बेटा! आज हम photosynthesis पढ़ेंगे।","format":"mp3"}' \
  -o hello.mp3
```

Each response carries `X-Request-Id`; a complete file also carries `X-Audio-Seconds`.

Streaming in JavaScript:
```js
const r = await fetch(`${BASE}/v1/tts`, {
  method: "POST",
  headers: { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({ voice_id, text, format: "pcm", sample_rate: 24000, stream: true }),
});
for await (const chunk of r.body) playPcm16(chunk);   // feed your audio player as chunks arrive
```

### `WS /v1/tts/stream` — speak text as it's written

Use this when text arrives piece by piece, for example from your own LLM.

```
wss://api.dhvni.in/v1/tts/stream?voice_id=voice_...&sample_rate=24000&quality=fast
```

| You send | Effect |
|---|---|
| `{"type":"text","text":"..."}` | add text; each complete sentence is spoken as soon as it's ready |
| `{"type":"flush"}` | speak whatever is buffered, even half a sentence |
| `{"type":"cancel"}` | stop speaking and drop anything queued |
| `{"type":"close"}` | finish what's queued, then close |

| You receive | Meaning |
|---|---|
| `{"type":"ready", "request_id", "sample_rate"}` | connected |
| binary frames | PCM16 little-endian mono audio, in order |
| `{"type":"chunk.done", "text", "audio_ms"}` | one piece finished |
| `{"type":"cancelled"}`, `{"type":"done", "characters", "audio_ms"}`, `{"type":"error", "message"}` | `characters` counts the text actually spoken; text dropped by a cancel is not billed |

The connection closes with code `4401` for a missing, wrong or expired key or token, `4403` for an account that isn't active, and `4400` for a bad `sample_rate` or `quality`.

## Speech-to-text

### `POST /v1/stt` (multipart form)

| Field | | |
|---|---|---|
| `file` | required | mp3, mp4, m4a, wav, webm, ogg or flac; up to 25 MB |
| `language` | optional | `hi` or `en`. Leave it out for mixed Hinglish |
| `prompt` | optional | names and terms to expect, e.g. `"NCERT, mitochondria, Ritesh Sir"` |
| `response_format` | `json` | `json` or `text` |

```bash
curl https://api.dhvni.in/v1/stt -H "Authorization: Bearer $KEY" -F file=@question.m4a
```
```json
{"text": "सर, mitochondria को powerhouse क्यों कहते हैं?", "model": "dhvni-transcribe-1", "duration_seconds": 3.2, "request_id": "req_..."}
```

## Usage

`GET /v1/usage?since=<unix>&until=<unix>` returns totals per metric. The default window is the last 30 days. Metrics are `tts_characters`, `tts_seconds` and `stt_seconds`.

## Errors and limits

Errors are JSON with a `detail` message: `{"detail": "..."}`.

| Status | When |
|---|---|
| 400 | bad input: unknown model, empty file, audio that can't be processed |
| 401 | missing, wrong or revoked API key |
| 402 | the account's credit balance is used up (only when prepaid credit is enforced) |
| 403 | the account is waiting for approval or suspended |
| 404 | the voice isn't on your account |
| 413 | text longer than 5,000 characters, or a file larger than 25 MB |
| 429 | too many requests (120 a minute) or too many at once (4 text-to-speech, 8 speech-to-text). Retry after the `Retry-After` header's seconds |
| 503 | temporarily busy; retry |
