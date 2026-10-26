"""Public v1 API: text-to-speech and speech-to-text.

Everything a customer sees here is branded (settings.brand_*). Engine names, model paths and
upstream error text never leave this module: failures map to neutral messages, details go to logs.

    GET  /v1/models            the two public models
    GET  /v1/voices            the customer's voices (operators clone them; see voicegw import-voice)
    POST /v1/tts               text -> audio file, or a chunked audio stream with "stream": true
    WS   /v1/tts/stream        text in as it's written (e.g. from an LLM), PCM16 audio out
    POST /v1/stt               audio file -> transcript
    POST /v1/tokens            short-lived token for browsers to open the WebSocket
"""

import asyncio
import contextlib
import json
import logging
import secrets
import struct
import time
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketDisconnect

from .audio import FORMATS, AudioError, StreamEncoder, encode, ffmpeg_convert, pcm_seconds
from .auth import CurrentCustomer, bearer, mint_session_token, read_session_token
from .config import settings
from .db import Customer
from .providers.transcribe import TranscribeError
from .providers.worker import TtsResult, WorkerError
from .text import SentenceSplitter

log = logging.getLogger(__name__)
router = APIRouter()

QUALITY_STEPS = {"fast": 8, "hq": 10}


def _rid() -> str:
    return "req_" + secrets.token_hex(8)


def _chunks(text: str) -> list[str]:
    """Long text -> sentence-sized pieces for the voice engine (one piece per render)."""
    sp = SentenceSplitter(first_min_chars=80, min_chars=120, max_chars=300)
    return [c for c in sp.feed(text.strip() + " ") + sp.flush() if c]


def _record(request_app, customer: Customer, rid: str, rows: list[tuple[str, float, str]]) -> None:
    request_app.state.db.add_usage([(customer.id, rid, kind, qty, provider, None) for kind, qty, provider in rows])


async def _voice_for(app, customer: Customer, voice_id: str) -> dict:
    v = app.state.db.voice_for(customer.id, voice_id)
    if v is None or v["status"] != "active":
        raise HTTPException(404, f"Voice {voice_id} not found on this account.")
    worker = app.state.worker
    try:
        if not await worker.has_voice(voice_id):
            await worker.put_voice(voice_id, (settings.voices_dir / voice_id / "reference.wav").read_bytes())
    except Exception as e:
        log.exception("could not load voice %s", voice_id)
        raise HTTPException(503, "The voice engine is temporarily unavailable. Please retry.") from e
    return v


async def _render(app, voice_id: str, text: str, sr: int, steps: int):
    """PCM16 chunks for one piece of text. Retries once if the engine lost the voice."""
    for attempt in range(2):
        result = TtsResult()
        try:
            async with contextlib.aclosing(app.state.worker.tts(voice_id, text, sr, result, steps)) as chunks:
                async for c in chunks:
                    yield c
            return
        except WorkerError as e:
            if e.code == "voice_not_found" and attempt == 0:
                await app.state.worker.put_voice(voice_id, (settings.voices_dir / voice_id / "reference.wav").read_bytes())
                continue
            raise


def _engine_error(e: Exception) -> HTTPException:
    if isinstance(e, WorkerError) and e.code in ("tts_busy", "tts_timeout"):
        return HTTPException(503, "The voice engine is busy. Please retry shortly.", headers={"Retry-After": "2"})
    log.exception("tts failed", exc_info=e)
    return HTTPException(500, "Speech generation failed. Please retry.")


# --- models -----------------------------------------------------------------------------------

@router.get("/v1/models", tags=["models"], summary="List models")
async def models(customer: Customer = CurrentCustomer):
    return {"models": [
        {"id": settings.brand_tts_model, "type": "text-to-speech", "languages": ["hi", "en", "hinglish"]},
        {"id": settings.brand_stt_model, "type": "speech-to-text", "languages": ["hi", "en", "hinglish"]},
    ]}


# --- text-to-speech ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    voice_id: str = Field(..., description="From GET /v1/voices.")
    text: str = Field(..., min_length=1, description="Hindi in Devanagari and English in Latin script reads best.")
    model: str | None = Field(None, description=f"Defaults to {settings.brand_tts_model}.")
    format: Literal["mp3", "opus", "wav", "pcm"] = Field("mp3", description="pcm = raw 16-bit little-endian mono.")
    sample_rate: Literal[16000, 22050, 24000, 44100, 48000] = 24000
    quality: Literal["fast", "hq"] = "fast"
    stream: bool = Field(False, description="Send audio as it's generated (chunked) instead of one file.")


@router.post("/v1/tts", tags=["text-to-speech"], summary="Text to speech",
             responses={200: {"content": {"audio/mpeg": {}, "audio/ogg": {}, "audio/wav": {}, "audio/pcm": {}}}})
async def tts(body: TTSRequest, request: Request, customer: Customer = CurrentCustomer):
    if body.model and body.model != settings.brand_tts_model:
        raise HTTPException(400, f"Unknown model {body.model!r}.")
    if len(body.text) > settings.tts_max_chars:
        raise HTTPException(413, f"Text is longer than {settings.tts_max_chars} characters.")
    app, rid = request.app, _rid()
    await _voice_for(app, customer, body.voice_id)
    pieces = _chunks(body.text)
    if not pieces:
        raise HTTPException(400, "Nothing to speak in that text.")
    steps, sr, fmt = QUALITY_STEPS[body.quality], body.sample_rate, body.format
    media_type = FORMATS[fmt][1]
    headers = {"X-Request-Id": rid}
    slot = app.state.limiter.acquire(customer.id, "tts")

    def bill(pcm_bytes: int) -> None:
        _record(app, customer, rid, [("tts_characters", len(body.text), "tts"),
                                     ("tts_seconds", pcm_bytes / (2 * sr), "tts")])

    if not body.stream:
        try:
            pcm = bytearray()
            for piece in pieces:
                async for c in _render(app, body.voice_id, piece, sr, steps):
                    pcm += c
            audio = await encode(bytes(pcm), sr, fmt)
        except (WorkerError, AudioError, OSError) as e:
            raise _engine_error(e) from e
        finally:
            app.state.limiter.release(slot)
        bill(len(pcm))
        headers["X-Audio-Seconds"] = f"{len(pcm) / (2 * sr):.2f}"
        return Response(audio, media_type=media_type, headers=headers)

    async def stream():
        total = 0
        try:
            if fmt in ("pcm", "wav"):
                if fmt == "wav":  # streaming WAV: sizes unknown up front, so the max value (players accept it)
                    yield _wav_header_unknown_size(sr)
                for piece in pieces:
                    async for c in _render(app, body.voice_id, piece, sr, steps):
                        total += len(c)
                        yield c
                return
            enc = StreamEncoder(sr, fmt)
            await enc.start()

            async def feed():
                nonlocal total
                try:
                    for piece in pieces:
                        async for c in _render(app, body.voice_id, piece, sr, steps):
                            total += len(c)
                            await enc.write(c)
                finally:
                    with contextlib.suppress(Exception):
                        await enc.finish()

            feeder = asyncio.create_task(feed())
            try:
                while data := await enc.read():
                    yield data
                await feeder
            finally:
                feeder.cancel()
                enc.kill()
        except (WorkerError, AudioError, OSError) as e:
            log.error("tts stream %s ended early: %r", rid, e)  # headers are sent; the stream just ends
        finally:
            app.state.limiter.release(slot)
            bill(total)

    return StreamingResponse(stream(), media_type=media_type, headers=headers)


def _wav_header_unknown_size(sr: int) -> bytes:
    data = 0xFFFFFFFF
    return (b"RIFF" + struct.pack("<I", data) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", data))


@router.post("/v1/tokens", tags=["auth"], summary="Short-lived token for browser WebSockets")
async def token(customer: Customer = CurrentCustomer):
    tok, exp = mint_session_token(customer.id)
    return {"token": tok, "expires_at": exp}


@router.websocket("/v1/tts/stream")
async def tts_stream(ws: WebSocket, voice_id: str = Query(...), sample_rate: int = Query(24000),
                     quality: str = Query("fast"), token: str | None = Query(None)):
    """Streaming TTS. Send {"type":"text","text":...} as text arrives, {"type":"flush"} to speak what's
    buffered, {"type":"cancel"} to stop, {"type":"close"} to finish and close. Audio comes back as
    binary PCM16 frames; {"type":"chunk.done", ...} after each piece and {"type":"done"} at the end."""
    db = ws.app.state.db
    key = bearer(ws.headers.get("authorization"))
    customer = db.customer_for_key(key) if key else (db.customer(read_session_token(token) or "") if token else None)
    if customer is None:
        await ws.close(code=4401, reason="unauthorized")
        return
    if sample_rate not in (16000, 22050, 24000, 44100, 48000) or quality not in QUALITY_STEPS:
        await ws.close(code=4400, reason="bad sample_rate or quality")
        return
    await ws.accept()
    app, rid = ws.app, _rid()
    try:
        await _voice_for(app, customer, voice_id)
        slot = app.state.limiter.acquire(customer.id, "tts")
    except HTTPException as e:
        await ws.send_json({"type": "error", "status": e.status_code, "message": e.detail})
        await ws.close()
        return
    steps = QUALITY_STEPS[quality]
    splitter = SentenceSplitter()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    chars = audio_bytes = 0
    send_lock = asyncio.Lock()

    async def send_json(obj):
        async with send_lock:
            await ws.send_json(obj)

    async def renderer():
        nonlocal audio_bytes
        while (piece := await queue.get()) is not None:
            n = 0
            async for c in _render(app, voice_id, piece, sample_rate, steps):
                n += len(c)
                async with send_lock:
                    await ws.send_bytes(c)
            audio_bytes += n
            await send_json({"type": "chunk.done", "text": piece, "audio_ms": round(n / (2 * sample_rate) * 1000)})
        await send_json({"type": "done", "characters": chars, "audio_ms": round(audio_bytes / (2 * sample_rate) * 1000)})

    task = asyncio.create_task(renderer())
    await send_json({"type": "ready", "request_id": rid, "model": settings.brand_tts_model, "sample_rate": sample_rate})
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            try:
                ev = json.loads(msg.get("text") or "{}")
            except json.JSONDecodeError:
                await send_json({"type": "error", "message": "Messages must be JSON."})
                continue
            kind = ev.get("type")
            if kind == "text":
                text = str(ev.get("text", ""))
                chars += len(text)
                for piece in splitter.feed(text):
                    queue.put_nowait(piece)
            elif kind == "flush":
                for piece in splitter.flush():
                    queue.put_nowait(piece)
            elif kind == "cancel":
                splitter = SentenceSplitter()
                while not queue.empty():
                    queue.get_nowait()
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                await send_json({"type": "cancelled"})
                task = asyncio.create_task(renderer())
            elif kind == "close":
                for piece in splitter.flush():
                    queue.put_nowait(piece)
                queue.put_nowait(None)
                with contextlib.suppress(Exception):
                    await task
                break
            else:
                await send_json({"type": "error", "message": f"Unknown message type {kind!r}."})
    except WebSocketDisconnect:
        pass
    except (WorkerError, AudioError, OSError) as e:
        log.error("tts ws %s failed: %r", rid, e)
        with contextlib.suppress(Exception):
            await send_json({"type": "error", "message": "Speech generation failed."})
    finally:
        task.cancel()
        app.state.limiter.release(slot)
        _record(app, customer, rid, [("tts_characters", chars, "tts"),
                                     ("tts_seconds", audio_bytes / (2 * sample_rate), "tts")])
        with contextlib.suppress(Exception):
            await ws.close()


# --- speech-to-text ---------------------------------------------------------------------------

@router.post("/v1/stt", tags=["speech-to-text"], summary="Speech to text")
async def stt(request: Request, file: UploadFile = File(..., description="mp3, mp4, m4a, wav, webm, ogg, flac (up to 25 MB)"),
              model: str | None = Form(None), language: str | None = Form(None, description="ISO-639-1, e.g. hi or en"),
              prompt: str | None = Form(None, description="Context or terms to expect (names, subject words)."),
              response_format: Literal["json", "text"] = Form("json"),
              customer: Customer = CurrentCustomer):
    if model and model != settings.brand_stt_model:
        raise HTTPException(400, f"Unknown model {model!r}.")
    app, rid = request.app, _rid()
    data = await file.read(settings.stt_max_bytes + 1)
    if len(data) > settings.stt_max_bytes:
        raise HTTPException(413, "Files are limited to 25 MB.")
    if not data:
        raise HTTPException(400, "The file is empty.")
    if not app.state.transcriber.available:
        raise HTTPException(503, "Transcription is temporarily unavailable.")
    async with app.state.limiter.slot(customer.id, "stt"):
        started = time.monotonic()
        try:
            result = await app.state.transcriber.transcribe(data, file.filename or "audio.wav", language, prompt)
        except TranscribeError as e:
            raise HTTPException(e.status, e.message, headers={"X-Request-Id": rid}) from None
    seconds = result.seconds
    if seconds is None:
        with contextlib.suppress(AudioError):
            seconds = round(pcm_seconds(await ffmpeg_convert(data, 16000, "s16le"), 16000), 2)
    _record(app, customer, rid, [("stt_seconds", seconds or 0, "stt")])
    log.info("stt %s: %.1fs audio in %.0f ms", rid, seconds or 0, (time.monotonic() - started) * 1000)
    headers = {"X-Request-Id": rid}
    if response_format == "text":
        return PlainTextResponse(result.text, headers=headers)
    return Response(json.dumps({"text": result.text, "model": settings.brand_stt_model,
                                "duration_seconds": seconds, "request_id": rid}, ensure_ascii=False),
                    media_type="application/json", headers=headers)
