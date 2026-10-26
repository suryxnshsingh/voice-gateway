"""Internal model API. Only the gateway talks to this; it is not customer-facing."""

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect

from .audio import decode, pcm16_to_float, write_wav
from .config import settings
from .tts import StreamStats, TTSBusy, TTSPool, VoiceNotFound

log = logging.getLogger("vgworker")

_VOICE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SAMPLE_RATES = {8000, 16000, 22050, 24000, 44100, 48000}


class State:
    tts: TTSPool | None = None
    tone = None
    speaker = None
    # Tone and speaker models are small and run on CPU; keep them off the event loop and off the TTS threads.
    cpu = ThreadPoolExecutor(max_workers=2, thread_name_prefix="cpu")


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.voices_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    if "tone" in settings.load:
        from .tone import ToneModel

        state.tone = await loop.run_in_executor(None, ToneModel, settings.tone_model, settings.tone_device)
        log.info("tone model ready")
    if "speaker" in settings.load:
        from .speaker import SpeakerModel

        state.speaker = await loop.run_in_executor(
            None, SpeakerModel, settings.speaker_model, settings.speaker_device, str(settings.data_dir / "ecapa")
        )
        log.info("speaker model ready")
    if "tts" in settings.load:
        state.tts = TTSPool(
            settings.tts_model,
            settings.tts_instances,
            settings.device,
            settings.voices_dir,
            settings.tts_cfg,
            settings.tts_timesteps,
        )
        await state.tts.warmup()
        log.info("tts warm")
        watchdog = asyncio.create_task(_watchdog(state.tts))
    yield


async def _watchdog(tts: TTSPool) -> None:
    while True:
        await asyncio.sleep(10)
        if settings.keep_warm_s and time.monotonic() - tts.last_used > settings.keep_warm_s:
            with contextlib.suppress(Exception):
                if await tts.keep_warm():
                    tts.last_used = time.monotonic()
        for op in tts.reclaim_leaked():
            log.error("reclaimed a leaked tts instance (held by %r after its thread finished)", op)
        stuck = [b for b in tts.busy() if b["for_s"] > 30]
        if stuck:
            log.warning("tts instance held for a long time: %s (kill -USR1 %d dumps stacks)", stuck, os.getpid())


app = FastAPI(title="voice-gateway worker", lifespan=lifespan)


def _authorized(token: str | None) -> bool:
    return not settings.token or (token is not None and secrets.compare_digest(token, settings.token))


@app.middleware("http")
async def auth(request: Request, call_next):
    if request.url.path != "/health" and not _authorized(request.headers.get("x-worker-token")):
        from fastapi.responses import JSONResponse

        return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


def _need(model, name: str):
    if model is None:
        raise HTTPException(503, f"{name} model not loaded on this worker")
    return model


def _check_voice_id(voice_id: str) -> None:
    if not _VOICE_ID.match(voice_id):
        raise HTTPException(400, "bad voice id")


@app.get("/health")
async def health():
    tts = state.tts
    return {
        "tts": None
        if tts is None
        else {"model": settings.tts_model, "instances": tts.instances, "idle": tts.idle, "busy": tts.busy(), "device": tts.device,
              "sample_rate": tts.sample_rate, "cached_voices": tts.cached_voices},
        "tone": state.tone is not None,
        "speaker": state.speaker is not None,
    }


@app.put("/voices/{voice_id}")
async def put_voice(voice_id: str, request: Request):
    """Store a reference recording (any soundfile-readable format) and warm its prompt cache."""
    _check_voice_id(voice_id)
    tts = _need(state.tts, "tts")
    data = await request.body()
    try:
        audio = decode(data, tts.sample_rate)
    except Exception as e:
        raise HTTPException(400, f"unreadable audio: {e}") from e
    write_wav(tts.voice_path(voice_id), audio, tts.sample_rate)
    await tts.load_voice(voice_id)
    return {"voice_id": voice_id, "seconds": round(len(audio) / tts.sample_rate, 2)}


@app.get("/voices/{voice_id}")
async def get_voice(voice_id: str):
    _check_voice_id(voice_id)
    tts = _need(state.tts, "tts")
    if not tts.voice_path(voice_id).exists():
        raise HTTPException(404, "voice not on this worker")
    cached = tts.is_cached(voice_id)
    if not cached:
        # Someone is about to use this voice (the gateway asks when a session picks it): warm it now.
        task = asyncio.create_task(tts.ensure_loaded(voice_id))
        task.add_done_callback(lambda t: not t.cancelled() and t.exception() and log.error("warming %s failed: %r", voice_id, t.exception()))
    return {"voice_id": voice_id, "cached": cached}


@app.delete("/voices/{voice_id}")
async def delete_voice(voice_id: str):
    _check_voice_id(voice_id)
    tts = _need(state.tts, "tts")
    tts.evict(voice_id)
    tts.voice_path(voice_id).unlink(missing_ok=True)
    return {"deleted": voice_id}


@app.post("/tone")
async def tone(request: Request):
    """Body: raw PCM16 LE mono at 16 kHz (one user utterance)."""
    model = _need(state.tone, "tone")
    audio = pcm16_to_float(await request.body())
    if len(audio) < 1600:
        raise HTTPException(400, "need at least 100 ms of audio")
    t0 = time.monotonic()
    result = await asyncio.get_running_loop().run_in_executor(state.cpu, model.classify, audio)
    result["infer_ms"] = round((time.monotonic() - t0) * 1000, 1)
    result["duration_ms"] = round(len(audio) / 16, 1)
    return result


@app.post("/speaker/similarity")
async def speaker_similarity(a: UploadFile = File(...), b: UploadFile = File(...)):
    model = _need(state.speaker, "speaker")
    try:
        xa = decode(await a.read(), 16000)
        xb = decode(await b.read(), 16000)
    except Exception as e:
        raise HTTPException(400, f"unreadable audio: {e}") from e
    score = await asyncio.get_running_loop().run_in_executor(state.cpu, model.similarity, xa, xb)
    return {"score": round(score, 4)}


@app.websocket("/tts")
async def tts_stream(ws: WebSocket):
    """One sentence per connection.

    Client sends {"voice_id", "text", "sample_rate"}; the worker streams binary PCM16 frames, then
    {"type": "done", ...stats}. Sending {"type": "cancel"} or closing the socket stops generation.
    """
    if not _authorized(ws.headers.get("x-worker-token")):
        await ws.close(code=4401)
        return
    await ws.accept()
    tts = state.tts
    if tts is None:
        await ws.send_json({"type": "error", "code": "tts_unavailable"})
        await ws.close()
        return
    try:
        req = json.loads(await ws.receive_text())
        voice_id, text = str(req["voice_id"]), str(req["text"]).strip()
        out_sr = int(req.get("sample_rate", 24000))
        # Optional quality knobs: fewer diffusion steps is faster; bounded so a caller can't stall the GPU.
        timesteps = int(req["inference_timesteps"]) if req.get("inference_timesteps") else None
        cfg_value = float(req["cfg_value"]) if req.get("cfg_value") else None
    except (KeyError, ValueError, TypeError):
        await ws.send_json({"type": "error", "code": "bad_request"})
        await ws.close()
        return
    if (not _VOICE_ID.match(voice_id) or not text or len(text) > settings.tts_max_chars or out_sr not in _SAMPLE_RATES
            or (timesteps is not None and not 2 <= timesteps <= 30) or (cfg_value is not None and not 0.5 <= cfg_value <= 5)):
        await ws.send_json({"type": "error", "code": "bad_request"})
        await ws.close()
        return

    cancelled = asyncio.Event()

    async def watch_cancel():
        with contextlib.suppress(WebSocketDisconnect, RuntimeError):
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect" or (msg.get("text") and json.loads(msg["text"]).get("type") == "cancel"):
                    cancelled.set()
                    return
        cancelled.set()

    watcher = asyncio.create_task(watch_cancel())
    stats = StreamStats()
    try:
        async with contextlib.aclosing(tts.stream(voice_id, text, out_sr, stats, timesteps=timesteps, cfg_value=cfg_value)) as chunks:
            async for chunk in chunks:
                if cancelled.is_set():
                    break
                await ws.send_bytes(chunk)
        status = "cancelled" if cancelled.is_set() else "done"
        with contextlib.suppress(Exception):
            await ws.send_json(
                {
                    "type": status,
                    "chars": len(text),
                    "audio_ms": round(stats.audio_ms, 1),
                    "queue_ms": round(stats.queue_ms, 1),
                    "first_chunk_ms": None if stats.first_chunk_ms is None else round(stats.first_chunk_ms, 1),
                    "total_ms": round(stats.total_ms, 1),
                }
            )
    except VoiceNotFound:
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "code": "voice_not_found"})
    except TTSBusy as e:
        log.error("%s", e)
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "code": "tts_busy"})
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("tts failed")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "code": "tts_failed"})
    finally:
        watcher.cancel()
        with contextlib.suppress(Exception):
            await ws.close()
