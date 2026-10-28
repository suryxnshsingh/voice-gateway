"""Public v1 API: text-to-speech and speech-to-text.

Everything a customer sees here is branded (settings.brand_*). Engine names, model paths and
upstream error text never leave this module: failures map to neutral messages, details go to logs.

    GET  /v1/models            the two public models
    GET  /v1/voices            the customer's voices (operators clone them; see voicegw import-voice)
    POST /v1/tts               text -> audio file, or a chunked audio stream with "stream": true
    WS   /v1/tts/stream        text in as it's written (e.g. from an LLM), PCM16 audio out
    POST /v1/stt               audio file -> transcript
    POST /v1/tokens            short-lived token for browsers to open the WebSocket

Keys, accounts and voices come from app.state.accounts (accounts.py). Every TTS/STT request that
gets past authentication leaves one row in the request log (Meter), whatever its outcome.
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
from .auth import CurrentCustomer, bearer, mint_session_token, read_session_claims
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


class Meter:
    """One request's row in the request log, written once when the request ends."""

    def __init__(self, app, customer: Customer, endpoint: str, rid: str, voice_id: str | None = None):
        self.app, self.customer, self.endpoint, self.rid, self.voice_id = app, customer, endpoint, rid, voice_id
        self.started = time.monotonic()
        self.characters = 0
        self.audio_seconds = 0.0
        self.latency_ms: int | None = None
        self._written = False

    def first_audio(self) -> None:
        if self.latency_ms is None:
            self.latency_ms = round((time.monotonic() - self.started) * 1000)

    def done(self, status: int = 200) -> None:
        if self._written:
            return
        self._written = True
        accounts = self.app.state.accounts
        self.app.state.db.add_request({
            "request_id": self.rid, "customer_id": self.customer.id, "api_key_id": self.customer.key_id,
            "endpoint": self.endpoint, "voice_id": self.voice_id, "characters": self.characters,
            "audio_seconds": self.audio_seconds, "status_code": status, "latency_ms": self.latency_ms,
            "created_at": time.time(),
        }, "pending" if accounts.remote else "none")

    def failed(self, e: HTTPException) -> HTTPException:
        """Logs the failure and returns it with X-Request-Id, for `raise meter.failed(e)`."""
        self.done(e.status_code)
        return HTTPException(e.status_code, e.detail, headers={**(e.headers or {}), "X-Request-Id": self.rid})


async def _check_credit(app, customer: Customer) -> None:
    if not settings.require_credit:
        return
    balance = await app.state.accounts.balance_paise(customer)
    if balance is not None and balance <= 0:
        raise HTTPException(402, "Your credit balance has run out. Add credits from the dashboard to continue.")


async def _voice_for(app, customer: Customer, voice_id: str) -> dict:
    v = await app.state.accounts.voice(customer, voice_id)
    if v is None:
        raise HTTPException(404, f"Voice {voice_id} not found on this account.")
    if not (settings.voices_dir / voice_id / "reference.wav").exists():
        log.error("voice %s is live for %s but this gateway has no reference audio for it", voice_id, customer.id)
        raise HTTPException(503, "This voice is not available right now. Please retry later.")
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
    app, rid = request.app, _rid()
    meter = Meter(app, customer, "tts", rid, body.voice_id)
    meter.characters = len(body.text)
    try:
        if body.model and body.model != settings.brand_tts_model:
            raise HTTPException(400, f"Unknown model {body.model!r}.")
        if len(body.text) > settings.tts_max_chars:
            raise HTTPException(413, f"Text is longer than {settings.tts_max_chars} characters.")
        await _voice_for(app, customer, body.voice_id)
        pieces = _chunks(body.text)
        if not pieces:
            raise HTTPException(400, "Nothing to speak in that text.")
        await _check_credit(app, customer)
        slot = app.state.limiter.acquire(customer.id, "tts")
    except HTTPException as e:
        raise meter.failed(e) from None
    steps, sr, fmt = QUALITY_STEPS[body.quality], body.sample_rate, body.format
    media_type = FORMATS[fmt][1]
    headers = {"X-Request-Id": rid}

    async def render(piece: str):
        async for c in _render(app, body.voice_id, piece, sr, steps):
            meter.first_audio()
            meter.audio_seconds += len(c) / (2 * sr)
            yield c

    if not body.stream:
        try:
            pcm = bytearray()
            for piece in pieces:
                async for c in render(piece):
                    pcm += c
            audio = await encode(bytes(pcm), sr, fmt)
        except (WorkerError, AudioError, OSError) as e:
            raise meter.failed(_engine_error(e)) from e
        finally:
            app.state.limiter.release(slot)
        meter.done()
        headers["X-Audio-Seconds"] = f"{len(pcm) / (2 * sr):.2f}"
        return Response(audio, media_type=media_type, headers=headers)

    async def stream():
        status = 499  # the client went away before the end; failed and abandoned streams cost nothing
        try:
            if fmt in ("pcm", "wav"):
                if fmt == "wav":  # streaming WAV: sizes unknown up front, so the max value (players accept it)
                    yield _wav_header_unknown_size(sr)
                for piece in pieces:
                    async for c in render(piece):
                        yield c
                status = 200
                return
            enc = StreamEncoder(sr, fmt)
            await enc.start()

            async def feed():
                try:
                    for piece in pieces:
                        async for c in render(piece):
                            await enc.write(c)
                finally:
                    with contextlib.suppress(Exception):
                        await enc.finish()

            feeder = asyncio.create_task(feed())
            try:
                while data := await enc.read():
                    yield data
                await feeder
                status = 200
            finally:
                feeder.cancel()
                enc.kill()
        except (WorkerError, AudioError, OSError) as e:
            status = 500
            log.error("tts stream %s ended early: %r", rid, e)  # headers are sent; the stream just ends
        finally:
            app.state.limiter.release(slot)
            meter.done(status)

    return StreamingResponse(stream(), media_type=media_type, headers=headers)


def _wav_header_unknown_size(sr: int) -> bytes:
    data = 0xFFFFFFFF
    return (b"RIFF" + struct.pack("<I", data) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16)
            + b"data" + struct.pack("<I", data))


@router.post("/v1/tokens", tags=["auth"], summary="Short-lived token for browser WebSockets")
async def token(customer: Customer = CurrentCustomer):
    tok, exp = mint_session_token(customer.id, key_id=customer.key_id)
    return {"token": tok, "expires_at": exp}


@router.websocket("/v1/tts/stream")
async def tts_stream(ws: WebSocket, voice_id: str = Query(...), sample_rate: int = Query(24000),
                     quality: str = Query("fast"), token: str | None = Query(None)):
    """Streaming TTS. Send {"type":"text","text":...} as text arrives, {"type":"flush"} to speak what's
    buffered, {"type":"cancel"} to stop, {"type":"close"} to finish and close. Audio comes back as
    binary PCM16 frames; {"type":"chunk.done", ...} after each piece and {"type":"done"} at the end."""
    async def refuse(code: int, reason: str) -> None:
        # Accept first: a close before the handshake completes reaches the client as a bare HTTP 403.
        await ws.accept()
        await ws.close(code=code, reason=reason[:120])

    accounts = ws.app.state.accounts
    key = bearer(ws.headers.get("authorization"))
    claims = read_session_claims(token) if token and not key else None
    try:
        customer = await accounts.for_key(key) if key else (await accounts.for_token(claims) if claims else None)
    except HTTPException as e:  # account not active, or accounts unreachable
        await refuse(4403 if e.status_code == 403 else 1013, str(e.detail))
        return
    if customer is None:
        await refuse(4401, "unauthorized")
        return
    if sample_rate not in (16000, 22050, 24000, 44100, 48000) or quality not in QUALITY_STEPS:
        await refuse(4400, "bad sample_rate or quality")
        return
    await ws.accept()
    app, rid = ws.app, _rid()
    meter = Meter(app, customer, "tts_stream", rid, voice_id)
    try:
        await _voice_for(app, customer, voice_id)
        await _check_credit(app, customer)
        slot = app.state.limiter.acquire(customer.id, "tts")
    except HTTPException as e:
        meter.done(e.status_code)
        await ws.send_json({"type": "error", "status": e.status_code, "message": e.detail})
        await ws.close()
        return
    steps = QUALITY_STEPS[quality]
    splitter = SentenceSplitter()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    audio_bytes = 0
    status = 200
    send_lock = asyncio.Lock()

    async def send_json(obj):
        async with send_lock:
            await ws.send_json(obj)

    async def renderer():
        # Billed per piece spoken, so text dropped by a cancel costs nothing.
        nonlocal audio_bytes, status
        try:
            while (piece := await queue.get()) is not None:
                n = 0
                async for c in _render(app, voice_id, piece, sample_rate, steps):
                    meter.first_audio()
                    n += len(c)
                    async with send_lock:
                        await ws.send_bytes(c)
                audio_bytes += n
                meter.characters += len(piece)
                meter.audio_seconds = audio_bytes / (2 * sample_rate)
                await send_json({"type": "chunk.done", "text": piece, "audio_ms": round(n / (2 * sample_rate) * 1000)})
            await send_json({"type": "done", "characters": meter.characters,
                             "audio_ms": round(audio_bytes / (2 * sample_rate) * 1000)})
        except Exception as e:  # anything, so the client hears about it instead of waiting forever
            status = 500
            log.error("tts ws %s failed: %r", rid, e, exc_info=not isinstance(e, (WorkerError, AudioError, OSError)))
            with contextlib.suppress(Exception):
                await send_json({"type": "error", "message": "Speech generation failed."})
                await ws.close(code=1011)

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
                if meter.latency_ms is None and not text.isspace():
                    meter.started = time.monotonic()  # latency: from the text that is waiting to be spoken
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
        status = 500
        log.error("tts ws %s failed: %r", rid, e)
        with contextlib.suppress(Exception):
            await send_json({"type": "error", "message": "Speech generation failed."})
    finally:
        task.cancel()
        app.state.limiter.release(slot)
        meter.done(status)
        with contextlib.suppress(Exception):
            await ws.close()


# --- speech-to-text ---------------------------------------------------------------------------

@router.post("/v1/stt", tags=["speech-to-text"], summary="Speech to text")
async def stt(request: Request, file: UploadFile = File(..., description="mp3, mp4, m4a, wav, webm, ogg, flac (up to 25 MB)"),
              model: str | None = Form(None), language: str | None = Form(None, description="ISO-639-1, e.g. hi or en"),
              prompt: str | None = Form(None, description="Context or terms to expect (names, subject words)."),
              response_format: Literal["json", "text"] = Form("json"),
              customer: Customer = CurrentCustomer):
    app, rid = request.app, _rid()
    meter = Meter(app, customer, "stt", rid)
    try:
        if model and model != settings.brand_stt_model:
            raise HTTPException(400, f"Unknown model {model!r}.")
        data = await file.read(settings.stt_max_bytes + 1)
        if len(data) > settings.stt_max_bytes:
            raise HTTPException(413, "Files are limited to 25 MB.")
        if not data:
            raise HTTPException(400, "The file is empty.")
        if not app.state.transcriber.available:
            raise HTTPException(503, "Transcription is temporarily unavailable.")
        await _check_credit(app, customer)
        async with app.state.limiter.slot(customer.id, "stt"):
            try:
                result = await app.state.transcriber.transcribe(data, file.filename or "audio.wav", language, prompt)
            except TranscribeError as e:
                raise HTTPException(e.status, e.message) from None
    except HTTPException as e:
        meter.latency_ms = round((time.monotonic() - meter.started) * 1000)
        raise meter.failed(e) from None
    seconds = result.seconds
    if seconds is None:
        with contextlib.suppress(AudioError):
            seconds = round(pcm_seconds(await ffmpeg_convert(data, 16000, "s16le"), 16000), 2)
    meter.audio_seconds = seconds or 0
    meter.latency_ms = round((time.monotonic() - meter.started) * 1000)
    meter.done()
    log.info("stt %s: %.1fs audio in %d ms", rid, seconds or 0, meter.latency_ms)
    headers = {"X-Request-Id": rid}
    if response_format == "text":
        return PlainTextResponse(result.text, headers=headers)
    return Response(json.dumps({"text": result.text, "model": settings.brand_stt_model,
                                "duration_seconds": seconds, "request_id": rid}, ensure_ascii=False),
                    media_type="application/json", headers=headers)
