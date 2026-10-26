"""One real-time voice session over a client WebSocket.

Input path:   mic PCM -> RNNoise -> Silero VAD -> barge-in controller
                 └-> Soniox STT (raw audio by default) -> words, end of turn
Output path:  LLM -> sentences -> VoxCPM2 -> Playout (server-paced audio buffer) -> client

Barge-in has two stages. VAD speech during a reply pauses playback at once; words from STT (or
long enough continuous speech) confirm it and cancel the reply. If the speech stops without any
real words (a cough, "haan", "okay"), playback resumes where it paused.

Client -> server
  binary                                   PCM16 LE mono 16 kHz microphone audio (20-100 ms frames)
  {"type": "session.update", "session": {...}}   see SessionConfig
  {"type": "response.create", "text": "..."}     speak this text (bring-your-own LLM); omit text to have the LLM reply
  {"type": "response.cancel"}
  {"type": "input_audio.commit"}                 end the user's turn now (push-to-talk)

Server -> client
  session.created / session.updated  {session}
  input.speech_started {source: vad | stt} / input.speech_stopped
  transcript.delta    {final, partial}
  transcript.done     {turn_id, text, start_ms, end_ms}        turn_id is null and "ignored" set for
                                                               backchannels and echo during a reply
  tone                {turn_id, source: "acoustic" | "fused", label, confidence, ...}
  response.started    {response_id, turn_id}
  response.text.delta {response_id, delta}
  response.audio.started {response_id, sample_rate}   binary frames that follow belong to this response
  binary              PCM16 LE mono at output_sample_rate, sent at playback speed
  output_audio.paused {response_id}        stop playing, keep what's queued
  output_audio.resumed {response_id, reason}
  response.done       {response_id, status: completed | cancelled | failed, text, reason, timings}
                      on cancelled, drop any audio still queued for playback
  error               {code, message}
"""

import asyncio
import base64
import contextlib
import difflib
import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

import numpy as np
from fastapi import WebSocket
from pydantic import BaseModel, Field, ValidationError
from starlette.websockets import WebSocketDisconnect

from . import bargein
from .audio import BYTES_PER_MS_IN, PcmRing
from .config import Settings
from .db import DB, Customer, new_id
from .frontend import AudioFrontend, VadEvent, VadParams, float_to_pcm16
from .playout import Playout
from .providers.llm import LLM, LlmUsage
from .providers import stt as stt_chain
from .providers.soniox import SttError, SttUpdate
from .providers.worker import TtsResult, WorkerClient, WorkerError
from .text import SentenceSplitter, speakable
from .tone import fuse
from .turn import SmartTurn

log = logging.getLogger(__name__)

TONE_MAX_MS = 6000

# Noise suppression and VAD for every session; onnxruntime and the RNNoise C code release the GIL.
_FRONTEND_POOL = ThreadPoolExecutor(max_workers=os.cpu_count() or 4, thread_name_prefix="frontend")

VOICE_RULES = """You are talking with the user in a live voice call. Everything you write is spoken aloud.
- Use short, natural spoken sentences: one to three unless the user asks for more detail.
- No markdown, lists, headings, emojis, URLs or code. Say numbers, symbols and formulas the way a person would.
- Reply in the language the user speaks: English, Hindi or Hinglish.
- Write Hindi and Hinglish in Devanagari, with English words and technical terms in English letters, e.g. \
"देखो, mitochondria cell का powerhouse है, क्योंकि यही ATP बनाता है।" Never write Hindi in Roman letters: the voice \
reads Devanagari much more naturally.
- A user message may end with [tone: ...], describing how the user sounded. Use it to adapt (slow down and check \
understanding when they are confused, reassure them when frustrated or anxious) but never mention it.
- If a message looks cut off or garbled by speech recognition, ask a short clarifying question instead of guessing.
- If the user interrupted you, your reply was cut where they heard it; don't repeat the part they already heard."""


class SessionConfig(BaseModel):
    voice_id: str | None = None
    instructions: str = Field("", max_length=8000)  # empty: the voice's own persona, if it has one
    greet: bool = False  # speak the voice's greeting as soon as the voice is set
    llm: bool = True          # false: the client runs its own LLM and sends response.create {text}
    model: str | None = None  # defaults to VG_LLM_MODEL
    reasoning_effort: str | None = None  # defaults to VG_LLM_REASONING_EFFORT; "none" for fastest replies
    tone: bool = True
    tone_fusion: bool = True  # also label tone from transcript + acoustic scores with a small LLM
    output_sample_rate: Literal[16000, 24000, 48000] = 24000
    language_hints: list[str] = ["en", "hi"]
    # input front end
    noise_suppression: bool = True
    denoise_stt: bool = False  # send denoised audio to STT too (can help in loud rooms, can hurt accuracy)
    vad: bool = True           # off: barge-in waits for STT words (VAD still runs for turn ends where STT needs it)
    vad_threshold: float = Field(0.6, ge=0.1, le=0.95)
    vad_start_ms: int = Field(200, ge=32, le=2000)
    vad_stop_ms: int = Field(500, ge=96, le=3000)
    # Silence that ends a turn, for STT providers without their own turn detection (OpenAI). Longer
    # than vad_stop_ms so a mid-sentence breath doesn't end the turn; Soniox decides this itself.
    turn_silence_ms: int = Field(800, ge=200, le=5000)
    # Start drafting the reply the moment the user pauses, from the live transcript, and hold it until
    # the turn is confirmed. Hides LLM + first-sentence TTS time behind end-of-turn detection; a draft
    # is discarded unseen if the user keeps talking or the final words differ.
    preemptive: bool = True
    # Semantic end-of-turn (Smart Turn v3.2): at a short pause, a confident "finished" ends the turn
    # right away; a confident "not finished" stretches the silence needed. See turn.py.
    smart_turn: bool = True
    # Pre-rendered openers ("अच्छा,", "देखो बेटा,"...) in the session's voice. When a reply starts with
    # one, its audio plays at once while the rest renders: hides TTS start-up time.
    openers: bool = True
    smart_turn_threshold: float = Field(0.5, ge=0.05, le=0.99)
    # barge-in
    # While the assistant is audible, speech must clear a higher bar to pause it: echo from speakers
    # reaches the mic quieter and blurrier than a user talking into it.
    barge_in_threshold: float = Field(0.85, ge=0.1, le=0.99)
    barge_in_start_ms: int = Field(300, ge=32, le=2000)
    interrupt_min_words: int = Field(1, ge=1, le=10)
    interrupt_confirm_ms: int = Field(800, ge=0, le=5000)  # sustained speech confirms before words arrive; 0 = words only
    false_interruption_ms: int = Field(700, ge=100, le=5000)  # after speech stops, resume if nothing real was said
    backchannels: bool = True  # "haan", "okay", "hmm" during a reply don't interrupt it
    echo_guard: bool = True    # ignore "speech" that is the assistant's own words coming back through the mic

    def vad_params(self, during_reply: bool = False) -> VadParams:
        threshold = self.barge_in_threshold if during_reply else self.vad_threshold
        return VadParams(threshold=threshold, stop_threshold=min(0.35, self.vad_threshold - 0.15),
                         start_ms=self.barge_in_start_ms if during_reply else self.vad_start_ms,
                         stop_ms=self.vad_stop_ms)


@dataclass
class Deps:
    db: DB
    worker: WorkerClient
    llm: LLM
    settings: Settings


@dataclass
class Turn:
    final: str = ""
    partial: str = ""
    start_ms: int | None = None
    end_ms: int | None = None
    speaking: bool = False


@dataclass
class Response:
    id: str
    turn_id: str | None
    user_msg: dict | None
    t0: float
    task: asyncio.Task | None = None
    out: Playout | None = None
    text: str = ""
    started: bool = False
    held: bool = False  # a draft: generating, but nothing reaches the client until the turn is confirmed
    confirmed: bool = False  # a draft whose turn was confirmed (it may still be draining held events)
    held_events: list = field(default_factory=list)
    draft_text: str | None = None
    record_history: bool = True
    status: str = "in_progress"
    cancel_reason: str | None = None
    paused: bool = False
    pauses: int = 0
    resume_task: asyncio.Task | None = None
    sentences: list[tuple[float, str]] = field(default_factory=list)  # (audio offset ms, text)
    audio_ms_queued: float = 0.0
    timings: dict = field(default_factory=dict)
    tts_chars: int = 0
    tts_audio_ms: float = 0.0

    def mark(self, name: str) -> None:
        self.timings.setdefault(name, round((time.monotonic() - self.t0) * 1000))

    @property
    def pending(self) -> bool:
        """An unconfirmed draft: may still be discarded without the client ever knowing."""
        return self.held and not self.confirmed

    @property
    def audible(self) -> bool:
        return self.out is not None and self.out.sent_ms > 0

    def spoken_text(self) -> str:
        """What the user heard: sentences whose audio had started playing."""
        if self.out is None:
            return ""
        played = self.out.played_ms()
        return " ".join(s for offset, s in self.sentences if offset < played)

    def queued_text(self) -> str:
        return " ".join(s for _, s in self.sentences)


async def _once(text: str) -> AsyncIterator[str]:
    yield text


def _same_words(a: str | None, b: str) -> bool:
    """Does the final transcript ask what the draft answered? Small STT revisions are fine."""
    wa, wb = bargein.words(a or ""), bargein.words(b)
    if not wa or not wb:
        return False
    return difflib.SequenceMatcher(None, wa, wb).ratio() >= 0.85


OPENERS = ["अच्छा सवाल है!", "देखो बेटा,", "हाँ बेटा,", "अच्छा,", "बिल्कुल,", "ठीक है,", "सुनो,", "हम्म,"]
OPENER_RULE = ("- Begin a reply with one of these short openers when it fits naturally: " + " | ".join(OPENERS)
               + ". Vary them; skip it if none fits.")

# (voice_id, sample_rate) -> {opener: pcm}. Rendered once per voice per process.
_OPENER_CACHE: dict[tuple[str, int], dict[str, bytes]] = {}
_OPENER_LOCKS: dict[tuple[str, int], asyncio.Lock] = {}


class Session:
    def __init__(self, ws: WebSocket, customer: Customer, deps: Deps):
        self.ws = ws
        self.customer = customer
        self.deps = deps
        self.s = deps.settings
        self.id = new_id("sess")
        self.cfg = SessionConfig()
        self.voice: dict | None = None
        self.stt = None  # SonioxStream | OpenAIRealtimeSTT
        self.frontend: AudioFrontend | None = None
        self.audio = PcmRing()
        self.history: list[dict] = []
        self._send_lock = asyncio.Lock()
        self._turn = Turn()
        self._turn_no = 0
        self._resp: Response | None = None
        self._carry = ""  # a turn whose reply was cut off before any audio; it joins the next turn
        self._vad_speaking = False
        self._vad_seen = False  # VAD has reported at least once, so _vad_speaking means something
        self._greeted = False
        self._turn_detector: SmartTurn | None = None
        self._speech_start_ms = 0
        self._turn_finalized = False  # the turn detector already ended this turn
        self._in_pause = False  # VAD hears a short silence inside speech (before any stop)
        self._stt_switching = False  # failing over to the next STT provider
        self._deferred_endpoint = False  # an endpoint arrived while the user was still talking
        self._closed = False
        self._last_reply: tuple[str, float] | None = None  # (text, when playout ended): its echo can trail
        self._turn_vad_start_ms: int | None = None  # VAD start of the current turn's first speech
        self._turn_p: float | None = None
        self._endpoint_task: asyncio.Task | None = None
        self._last_turn_end_ms = 0
        self._tasks: set[asyncio.Task] = set()
        self._usage: list[tuple] = []
        self._llm_provider = urlparse(self.s.llm_base_url).hostname if self.s.llm_base_url else "openai"

    # plumbing

    async def send(self, event: dict) -> None:
        async with self._send_lock:
            await self.ws.send_json(event)

    async def _safe_send(self, event: dict) -> None:
        with contextlib.suppress(Exception):
            await self.send(event)

    async def _send_audio(self, pcm: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(pcm)

    async def error(self, code: str, message: str) -> None:
        await self._safe_send({"type": "error", "code": code, "message": message})

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _record(self, kind: str, quantity: float, provider: str, model: str | None) -> None:
        self._usage.append((self.customer.id, self.id, kind, quantity, provider, model))

    def _flush_usage(self) -> None:
        if self._usage:
            rows, self._usage = self._usage, []
            self.deps.db.add_usage(rows)

    # lifecycle

    async def run(self) -> None:
        await self.send({"type": "session.created", "session_id": self.id, "session": self.cfg.model_dump()})
        try:
            await self._read_client()
        except WebSocketDisconnect:
            pass
        finally:
            await self._shutdown()

    async def _shutdown(self) -> None:
        self._closed = True
        await self._interrupt("session_closed")
        for t in list(self._tasks):
            t.cancel()
        if self.stt is not None:
            await self._close_stt(self.stt)
        if self.frontend is not None:
            self.frontend.close()
        self._flush_usage()

    async def _read_client(self) -> None:
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                await self._on_audio(msg["bytes"])
            elif msg.get("text"):
                try:
                    event = json.loads(msg["text"])
                except json.JSONDecodeError:
                    await self.error("bad_event", "not JSON")
                    continue
                await self._on_event(event)

    # client events

    async def _on_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "session.update":
            await self._update(event.get("session") or {})
        elif kind == "response.create":
            await self._interrupt("replaced")
            text = event.get("text")
            if text is None and not self.deps.llm.available:
                await self.error("llm_unavailable", "no LLM configured; send text to speak")
                return
            self._start_response(None, None, text=str(text) if text is not None else None)
        elif kind == "response.cancel":
            await self._interrupt("client")
        elif kind == "input_audio.commit":
            if self.stt is not None:
                await self.stt.finalize()
        else:
            await self.error("bad_event", f"unknown event type {kind!r}")

    async def _update(self, patch: dict) -> None:
        try:
            cfg = SessionConfig.model_validate({**self.cfg.model_dump(), **patch})
        except ValidationError as e:
            await self.error("bad_session", e.errors()[0]["msg"])
            return
        if cfg.voice_id != self.cfg.voice_id:
            voice = self.deps.db.voice_for(self.customer.id, cfg.voice_id) if cfg.voice_id else None
            if cfg.voice_id and (voice is None or voice["status"] != "active"):
                await self.error("voice_unavailable", f"voice {cfg.voice_id} is not an active voice on this account")
                return
            self.voice = voice
            if voice:
                self._spawn(self._ensure_voice_on_worker())
        if self.stt is not None and cfg.language_hints != self.cfg.language_hints:
            await self.error("ignored", "language_hints only apply before the first audio frame")
            cfg.language_hints = self.cfg.language_hints
        if self.frontend is not None:
            if cfg.noise_suppression != self.cfg.noise_suppression:
                self.frontend.close()
                self.frontend = None  # rebuilt on the next frame
            else:
                self.frontend.params = cfg.vad_params()
        greet = cfg.greet and not self._greeted and self.voice is not None and self.voice.get("greeting")
        self.cfg = cfg
        await self.send({"type": "session.updated", "session": cfg.model_dump()})
        if greet:
            self._greeted = True
            self.history.append({"role": "assistant", "content": self.voice["greeting"]})
            self._start_response(None, None, text=self.voice["greeting"], record_history=False)

    async def _ensure_voice_on_worker(self) -> None:
        voice_id = self.voice["id"]
        try:
            if not await self.deps.worker.has_voice(voice_id):
                await self._push_voice(voice_id)
            if self.cfg.openers and (voice_id, self.cfg.output_sample_rate) not in _OPENER_CACHE:
                # Don't make the greeting (or a first reply) queue behind eight opener renders.
                for _ in range(150):
                    if self._resp is None:
                        break
                    await asyncio.sleep(0.2)
                await self._render_openers(voice_id, self.cfg.output_sample_rate)
        except Exception:
            log.exception("could not pre-load voice %s", voice_id)

    def _openers_ready(self) -> bool:
        return bool(self.cfg.openers and self.voice and _OPENER_CACHE.get((self.voice["id"], self.cfg.output_sample_rate)))

    async def _render_openers(self, voice_id: str, sr: int) -> None:
        key = (voice_id, sr)
        lock = _OPENER_LOCKS.setdefault(key, asyncio.Lock())
        async with lock:
            if key in _OPENER_CACHE:
                return
            # On disk too, so a gateway restart doesn't re-render them while the first user is talking.
            # Keyed by the reference recording and the worker's model: a new voice or fine-tune re-renders.
            ref = self.s.voices_dir / voice_id / "reference.wav"
            model = (await self.deps.worker.health()).get("tts", {}).get("model", "")
            tag = hashlib.sha1(f"{ref.stat().st_mtime_ns}:{ref.stat().st_size}:{model}:{'|'.join(OPENERS)}".encode()).hexdigest()[:12]
            path = self.s.voices_dir / voice_id / f"openers_{sr}_{tag}.json"
            if path.exists():
                data = json.loads(path.read_text())
                _OPENER_CACHE[key] = {k: base64.b64decode(v) for k, v in data.items()}
                return
            clips = {}
            for opener in OPENERS:
                result, pcm = TtsResult(), bytearray()
                async with contextlib.aclosing(self.deps.worker.tts(voice_id, opener, sr, result)) as chunks:
                    async for chunk in chunks:
                        pcm += chunk
                clips[opener] = bytes(pcm)
            path.write_text(json.dumps({k: base64.b64encode(v).decode() for k, v in clips.items()}, ensure_ascii=False))
            _OPENER_CACHE[key] = clips
            log.info("rendered %d openers for %s at %d Hz", len(clips), voice_id, sr)

    async def _push_voice(self, voice_id: str) -> None:
        ref = (self.s.voices_dir / voice_id / "reference.wav").read_bytes()
        await self.deps.worker.put_voice(voice_id, ref)

    # audio in -> front end + STT

    async def _on_audio(self, pcm: bytes) -> None:
        if len(pcm) % 2:
            pcm = pcm[:-1]
        if self.stt is None and not self._stt_switching:
            try:
                await self._connect_stt()
            except SttError as e:
                await self.error("stt_unavailable", str(e))
                await self.ws.close(code=1011)
                return
        if self.frontend is None:
            self.frontend = AudioFrontend(self.cfg.vad_params(), noise_suppression=self.cfg.noise_suppression)

        denoise_stt = self.frontend is not None and self.cfg.denoise_stt and self.cfg.noise_suppression
        if not denoise_stt:
            # STT first: it's on the critical path for end-of-turn; the front end can trail by a millisecond.
            self.audio.append(pcm)
            await self._send_stt(pcm)
        if self.frontend is None:
            return
        frontend = self.frontend
        frontend.params = self.cfg.vad_params(during_reply=self._resp is not None and self._resp.audible)
        res = await asyncio.get_running_loop().run_in_executor(_FRONTEND_POOL, frontend.process, pcm)
        if denoise_stt:
            clean = float_to_pcm16(res.denoised)
            self.audio.append(clean)
            await self._send_stt(clean)
        for ev in res.events:
            await self._on_vad(ev)
        if res.speaking:
            await self._on_vad_speaking(res.speech_ms)

    async def _send_stt(self, pcm: bytes) -> None:
        stream = self.stt
        if stream is None:
            return  # failing over: the ring buffer has this frame and the replay will send it
        try:
            await stream.send_audio(pcm)
        except Exception:
            pass  # the stream died; its reader raises, fails over and replays from the ring buffer

    async def _connect_stt(self, replay_from_ms: int | None = None) -> None:
        stream = await stt_chain.connect(self.s, self.cfg.language_hints)
        if self.stt is not None:  # someone else connected meanwhile; keep theirs
            await stream.close()
            return
        # Token times from this stream count from where it joined the session's audio timeline.
        stream.offset_ms = self.audio.end_ms if replay_from_ms is None else replay_from_ms
        self.stt = stream
        self._spawn(self._read_stt(stream))
        if replay_from_ms is not None:
            await stream.send_audio(self.audio.slice(replay_from_ms, self.audio.end_ms))
        await self._safe_send({"type": "stt.connected", "provider": stream.provider, "model": stream.model})

    async def _close_stt(self, stream) -> None:
        self._record("stt_seconds", stream.audio_bytes / (2 * 16000), stream.provider, stream.model)
        await stream.close()

    async def _read_stt(self, stream) -> None:
        try:
            async for up in stream.updates():
                if stream is not self.stt:
                    return
                await self._on_stt(up, stream.offset_ms)
            if stream is self.stt and not self._closed:
                # Every adapter should raise on a close it didn't ask for; this catches one that doesn't.
                raise SttError(f"{stream.provider} stream ended")
        except SttError as e:
            if stream is not self.stt:
                return
            log.error("session %s: %s stt failed: %s", self.id, stream.provider, e)
            stt_chain.mark_down(stream.provider, e)
            self._stt_switching = True
            self.stt = None
            self._spawn(self._close_stt(stream))  # a dead socket can take seconds to close; don't wait
            # Fail over without losing words: re-send the current turn's audio to the next provider.
            t = self._turn
            replay = (t.start_ms - 300) if t.start_ms is not None else self._last_turn_end_ms
            replay = max(replay, self.audio.end_ms - 15000, 0)
            self._turn = Turn()
            try:
                await self._connect_stt(replay_from_ms=replay)
            except SttError as e2:
                await self.error("stt_failed", str(e2))
                with contextlib.suppress(Exception):
                    await self.ws.close(code=1011)
            finally:
                self._stt_switching = False

    # barge-in controller

    async def _on_vad(self, ev: VadEvent) -> None:
        self._vad_seen = True
        if ev.kind in ("start", "resume") and self._resp is not None and self._resp.pending:
            await self._interrupt("draft_discarded")  # they kept talking: the draft answered half a question
        r = self._resp if self.cfg.vad else None
        if ev.kind == "pause":
            self._in_pause = True
            if self.cfg.smart_turn and self.stt is not None and (self._resp is None or self._resp.pending):
                self._spawn(self._check_turn_end(self.stt))
            return
        if ev.kind in ("start", "resume"):
            if ev.kind == "start":
                self._vad_speaking = True
                self._speech_start_ms = self.audio.end_ms
                if self._turn_vad_start_ms is None:
                    self._turn_vad_start_ms = self.audio.end_ms
                if self.cfg.llm and self.deps.llm.available:
                    self._spawn(self.deps.llm.warm(self.cfg.model or self.s.llm_model))  # reply comes soon: warm up
                await self.send({"type": "input.speech_started", "source": "vad"})
            self._in_pause = False
            # Whatever Smart Turn concluded at the pause no longer holds: they're talking again.
            self._turn_finalized = False
            self._turn_p = None
            if self._endpoint_task is not None:
                self._endpoint_task.cancel()  # same turn
            if r is not None and r.resume_task is not None:
                r.resume_task.cancel()  # a pending "false alarm" resume must not fire mid-speech
            if r is not None and r.audible and not r.paused:
                # Stage 1: stop talking over the user right away; decide whether it's real in a moment.
                r.out.pause()
                r.paused = True
                r.pauses += 1
                r.mark("paused_for_speech")
                await self.send({"type": "output_audio.paused", "response_id": r.id})
        elif ev.kind == "stop":
            self._vad_speaking = False
            self._in_pause = False
            await self.send({"type": "input.speech_stopped", "speech_ms": ev.speech_ms})
            if self.stt is not None and self.stt.needs_endpointing:
                # This STT has no turn detection of its own: end the turn after enough silence.
                self._endpoint_task = self._spawn(self._endpoint_after_silence(self.stt))
            if r is not None and r.paused:
                if r.resume_task is not None:
                    r.resume_task.cancel()
                r.resume_task = self._spawn(self._resume_if_false_alarm(r))
            if self._deferred_endpoint and not (self.stt is not None and self.stt.needs_endpointing):
                await self._handle_endpoint()
                return
            self._maybe_draft()

    def _maybe_draft(self) -> None:
        if not (self.cfg.preemptive and self.cfg.llm and self.deps.llm.available) or self._resp is not None:
            return
        heard = (self._turn.final + self._turn.partial).strip()
        if not heard or self._is_recent_echo(heard):
            return
        text = " ".join(x for x in (self._carry, heard) if x)
        user_msg = {"role": "user", "content": text}
        resp = Response(new_id("resp"), None, user_msg, time.monotonic(), held=True, draft_text=text)
        self._resp = resp
        resp.task = self._spawn(self._response_main(resp, None, None, pending_user=user_msg))

    def _is_recent_echo(self, text: str) -> bool:
        """The client still plays the last ~lead_ms of a reply after the server counts it done, and STT
        trails by ~1 s, so the mic's echo of a reply's last words can surface as a "turn" just after."""
        if not (self.cfg.echo_guard and self._last_reply):
            return False
        said, ended = self._last_reply
        return (time.monotonic() - ended < 2.5 and len(bargein.words(text)) >= 3
                and bargein.looks_like_echo(text, said))

    async def _emit(self, resp: Response, event: dict) -> None:
        if resp.held:
            resp.held_events.append(event)
        else:
            await self.send(event)

    async def _release_draft(self, resp: Response, turn_id: str) -> None:
        resp.confirmed = True  # from here on it's a real reply: never discarded silently
        resp.turn_id = turn_id
        resp.mark("turn_confirmed")
        # Drain under the send lock so nothing (e.g. this reply's own response.done) can jump ahead;
        # anything emitted meanwhile lands in the same list and is drained too.
        async with self._send_lock:
            while resp.held_events:
                ev = resp.held_events.pop(0)
                if ev["type"] == "response.started":
                    ev["turn_id"] = turn_id
                await self.ws.send_json(ev)
            resp.held = False
        if resp.out is not None:
            resp.out.resume()

    async def _endpoint_after_silence(self, stream) -> None:
        silence = self.cfg.turn_silence_ms
        if self._turn_p is not None and self._turn_p < 0.15:
            silence = max(silence, 1200)  # they sound mid-thought: give the pause room
        await asyncio.sleep(max(silence - self.cfg.vad_stop_ms, 0) / 1000)
        if stream is self.stt and not self._vad_speaking and not self._turn_finalized:
            await stream.finalize()

    async def _check_turn_end(self, stream) -> None:
        """At a short pause: does the speaker sound finished? If clearly yes, end the turn now."""
        loop = asyncio.get_running_loop()
        if self._turn_detector is None:
            self._turn_detector = await loop.run_in_executor(_FRONTEND_POOL, SmartTurn)  # loads the ONNX once
        end = self.audio.end_ms
        pcm = self.audio.slice(max(self._speech_start_ms - 300, end - 8000, 0), end)
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        p = await loop.run_in_executor(_FRONTEND_POOL, self._turn_detector.p_complete, audio)
        self._turn_p = p
        if p < self.cfg.smart_turn_threshold or not self._quiet() or stream is not self.stt:
            return
        self._turn_finalized = True
        if self._endpoint_task is not None:
            self._endpoint_task.cancel()
        await stream.finalize()
        if self._quiet():  # they may have resumed while the commit was in flight
            self._maybe_draft()

    def _quiet(self) -> bool:
        return self._in_pause or not self._vad_speaking

    async def _on_vad_speaking(self, speech_ms: int) -> None:
        r = self._resp if self.cfg.vad else None
        if r is not None and r.paused and self.cfg.interrupt_confirm_ms and speech_ms >= self.cfg.interrupt_confirm_ms:
            await self._interrupt("barge_in")

    async def _resume_if_false_alarm(self, r: Response) -> None:
        await asyncio.sleep(self.cfg.false_interruption_ms / 1000)
        if self._resp is r and r.paused and not self._vad_speaking:
            await self._resume(r, "false_interruption")

    async def _resume(self, r: Response, reason: str) -> None:
        r.paused = False
        r.out.resume()
        await self.send({"type": "output_audio.resumed", "response_id": r.id, "reason": reason})

    def _confirms_interruption(self, heard: str, r: Response) -> bool:
        real = bargein.real_words(heard, self.cfg.backchannels)
        if len(real) < self.cfg.interrupt_min_words:
            return False
        if self.cfg.echo_guard and r.audible and bargein.looks_like_echo(heard, r.queued_text()):
            return False
        return True

    # STT

    async def _on_stt(self, up: SttUpdate, offset_ms: int = 0) -> None:
        t = self._turn
        for tok in up.finals:
            if t.start_ms is None and tok.text.strip() and tok.start_ms is not None:
                t.start_ms = tok.start_ms + offset_ms
            if tok.end_ms is not None:
                t.end_ms = tok.end_ms + offset_ms
            t.final += tok.text
        t.partial = up.partial
        heard = (t.final + t.partial).strip()
        if heard:
            if not t.speaking:
                t.speaking = True
                if self.frontend is None:
                    await self.send({"type": "input.speech_started", "source": "stt"})
            r = self._resp
            if r is not None and not r.pending and self._confirms_interruption(heard, r):
                await self._interrupt("barge_in")
        if up.finals or up.partial:
            await self.send({"type": "transcript.delta", "final": t.final.strip(), "partial": t.partial})
        if heard and not up.endpoint and self._vad_seen and not self._vad_speaking:
            # STT words trail speech by up to a second, so they can land after the pause was heard:
            # draft now if nothing was drafted, or redraft if the draft missed words.
            r = self._resp
            if r is None:
                self._maybe_draft()
            elif r.pending and len(bargein.words(heard)) > len(bargein.words(r.draft_text)) - len(bargein.words(self._carry)):
                await self._interrupt("draft_stale")
                self._maybe_draft()
        if up.endpoint:
            if t.final.strip() and self._vad_seen and self._vad_speaking and not self._in_pause:
                # They're talking again: this final closed an earlier stretch. Keep the turn open; OpenAI's
                # next commit ends it, and for Soniox the next VAD stop does (it won't send another <end>).
                self._deferred_endpoint = True
                return
            await self._handle_endpoint()

    async def _handle_endpoint(self) -> None:
        self._deferred_endpoint = False
        t = self._turn
        text = t.final.strip()
        r = self._resp
        if text and r is None and self._is_recent_echo(text):
            self._turn = Turn()
            await self.send({"type": "transcript.done", "turn_id": None, "text": text, "ignored": "echo",
                             "start_ms": t.start_ms, "end_ms": t.end_ms})
            return
        if text and r is not None and not r.pending and self._ignorable_during_reply(text, r):
            self._turn = Turn()
            kind = "backchannel" if bargein.is_backchannel(text) else "echo"
            await self.send({"type": "transcript.done", "turn_id": None, "text": text, "ignored": kind,
                             "start_ms": t.start_ms, "end_ms": t.end_ms})
            if r.paused:
                await self._resume(r, kind)
        elif text:
            await self._end_turn()
        else:
            self._turn = Turn()

    def _ignorable_during_reply(self, text: str, r: Response) -> bool:
        if self.cfg.backchannels and bargein.is_backchannel(text):
            return True
        return self.cfg.echo_guard and r.audible and bargein.looks_like_echo(text, r.queued_text())

    # turns

    async def _end_turn(self) -> None:
        t, self._turn = self._turn, Turn()
        vad_start, self._turn_vad_start_ms = self._turn_vad_start_ms, None
        if t.end_ms is not None:
            self._last_turn_end_ms = t.end_ms
        t0 = time.monotonic()
        text = " ".join(x for x in (self._carry, t.final.strip()) if x)
        self._carry = ""
        self._turn_no += 1
        turn_id = f"turn_{self._turn_no}"
        await self.send({"type": "transcript.done", "turn_id": turn_id, "text": text, "start_ms": t.start_ms, "end_ms": t.end_ms})
        draft = self._resp if self._resp is not None and self._resp.pending else None
        if draft is not None and _same_words(draft.draft_text, text):
            user_msg = draft.user_msg
            user_msg["content"] = text
            self.history.append(user_msg)
            self._start_tone(t, turn_id, text, user_msg, vad_start)
            await self._release_draft(draft, turn_id)
            return
        await self._interrupt("new_turn")
        user_msg = {"role": "user", "content": text}
        self.history.append(user_msg)

        acoustic = self._start_tone(t, turn_id, text, user_msg, vad_start)

        if not self.cfg.llm or not self.deps.llm.available:
            return

        async def wait_for_tone(resp: Response) -> None:
            # Optional: give the reply this turn's acoustic tone if it's quick. Off by default; every
            # millisecond here is dead air, and later turns get the tone anyway.
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(acoustic), self.s.tone_wait_ms / 1000)
            resp.mark("tone_ready" if acoustic.done() else "tone_skipped")

        self._start_response(turn_id, user_msg, pre=wait_for_tone if self.s.tone_wait_ms > 0 else None, t0=t0)

    def _start_tone(self, t: Turn, turn_id: str, text: str, user_msg: dict, vad_start: int | None) -> asyncio.Future:
        acoustic: asyncio.Future = asyncio.get_running_loop().create_future()
        if self.cfg.tone and t.start_ms is not None and t.end_ms is not None:
            # The last few seconds carry the tone; emotion2vec's cost grows with length (~1.6 s CPU for 14 s).
            start = max(t.start_ms - 50, t.end_ms - TONE_MAX_MS, 0)
            if vad_start is not None:
                # OpenAI turns start at the previous commit, which may be silence or the assistant's echo.
                start = max(start, vad_start - 300)
            pcm = self.audio.slice(start, t.end_ms + 150)
            self._spawn(self._tone(turn_id, text, pcm, user_msg, acoustic))
        else:
            acoustic.set_result(None)
        return acoustic

    async def _tone(self, turn_id: str, text: str, pcm: bytes, user_msg: dict, acoustic: asyncio.Future) -> None:
        result = None
        try:
            if len(pcm) >= 100 * BYTES_PER_MS_IN:
                result = await self.deps.worker.tone(pcm)
                self._record("tone_calls", 1, "emotion2vec", None)
                user_msg["tone"] = result["label"]
                await self._safe_send({"type": "tone", "turn_id": turn_id, "source": "acoustic", **result})
        except Exception:
            log.exception("acoustic tone failed")
        finally:
            if not acoustic.done():
                acoustic.set_result(result)
        if not (self.cfg.tone_fusion and self.deps.llm.available):
            return
        try:
            usage = LlmUsage(self.s.tone_model)
            fused = await fuse(self.deps.llm, self.s.tone_model, text, result, usage)
            self._record_llm(usage)
            # Later turns see the fused label; a reply already in flight used the acoustic one.
            user_msg["tone"] = fused["label"]
            await self._safe_send({"type": "tone", "turn_id": turn_id, "source": "fused", **fused})
        except Exception:
            log.exception("tone fusion failed")

    # responses

    def _messages(self, pending_user: dict | None = None) -> list[dict]:
        persona = self.cfg.instructions or (self.voice or {}).get("persona") or ""
        rules = VOICE_RULES + ("\n" + OPENER_RULE if self._openers_ready() else "")
        system = rules + (f"\n\n{persona}" if persona else "")
        msgs = [{"role": "system", "content": system}]
        turns = self.history[-2 * self.s.history_turns :] + ([pending_user] if pending_user else [])
        for m in turns:
            content = m["content"]
            if m["role"] == "user" and m.get("tone"):
                content += f"\n[tone: {m['tone']}]"
            msgs.append({"role": m["role"], "content": content})
        return msgs

    def _start_response(self, turn_id, user_msg, *, text: str | None = None, pre=None, t0: float | None = None,
                        record_history: bool = True) -> Response:
        self._drop_current("replaced")
        resp = Response(new_id("resp"), turn_id, user_msg, t0 or time.monotonic(), record_history=record_history)
        self._resp = resp
        resp.task = self._spawn(self._response_main(resp, text, pre))
        return resp

    def _drop_current(self, reason: str) -> None:
        """Cancel the current reply without waiting, before another takes its place."""
        r = self._resp
        if r is not None and r.task is not None and not r.task.done():
            r.cancel_reason = reason
            if r.out is not None:
                r.out.pause()
            r.task.cancel()
        self._resp = None

    async def _interrupt(self, reason: str) -> None:
        resp = self._resp
        if resp is None or resp.task is None or resp.task.done():
            return
        self._resp = None
        resp.cancel_reason = reason
        if not resp.audible and resp.user_msg is not None and reason == "barge_in":
            # The user kept talking before hearing anything: treat both fragments as one turn.
            for i, m in enumerate(self.history):
                if m is resp.user_msg:
                    del self.history[i]
                    self._carry = resp.user_msg["content"]
                    break
        if resp.out is not None:
            resp.out.pause()  # freeze the playback clock now, so "what was heard" is exact
        resp.task.cancel()
        await asyncio.wait({resp.task}, timeout=2)

    async def _response_main(self, resp: Response, text: str | None, pre, pending_user: dict | None = None) -> None:
        model = self.cfg.model or self.s.llm_model
        usage = LlmUsage(model)
        sr = self.cfg.output_sample_rate

        async def first_frame() -> None:
            resp.mark("first_audio")
            if "turn_confirmed" in resp.timings:
                resp.timings["first_audio_after_turn_end"] = resp.timings["first_audio"] - resp.timings["turn_confirmed"]
            await self._emit(resp, {"type": "response.audio.started", "response_id": resp.id, "sample_rate": sr})

        try:
            if pre is not None:
                await pre(resp)
            await self._emit(resp, {"type": "response.started", "response_id": resp.id, "turn_id": resp.turn_id})
            resp.started = True
            resp.out = Playout(self._send_audio, sr, lead_ms=self.s.playout_lead_ms, on_first_frame=first_frame,
                               prebuffer_ms=self.s.playout_prebuffer_ms)
            if resp.held:
                resp.out.pause()  # render ahead, play nothing until the turn is confirmed
            if text is not None:
                deltas = _once(text)
            else:
                effort = self.cfg.reasoning_effort or self.s.llm_reasoning_effort or None
                deltas = self.deps.llm.stream(model, self._messages(pending_user), self.s.llm_max_tokens, usage, effort)
            await self._pipeline(resp, deltas)
            # Everything is rendered; the reply is still live (and interruptible) until it has been played.
            resp.out.end()
            await resp.out.drained.wait()
            resp.status = "completed"
        except asyncio.CancelledError:
            if resp.cancel_reason is None:
                raise
            resp.status = "cancelled"
        except Exception as e:
            log.exception("response failed")
            resp.status = "failed"
            await self.error("response_failed", str(e))
        finally:
            if self._resp is resp:
                self._resp = None
            if resp.out is not None:
                await resp.out.close()
                if resp.out.underruns:
                    resp.timings["underruns"] = resp.out.underruns
                    resp.timings["underrun_ms"] = round(resp.out.underrun_ms)
            if resp.resume_task is not None:
                resp.resume_task.cancel()
            said = speakable(resp.text if resp.status == "completed" else resp.spoken_text())
            if said and not resp.pending:
                self._last_reply = (resp.queued_text() or said, time.monotonic())
            if said and resp.record_history:
                self.history.append({"role": "assistant", "content": said})
            if text is None:
                self._record_llm(usage)
            self._record("tts_characters", resp.tts_chars, "voxcpm2", None)
            self._record("tts_seconds", resp.tts_audio_ms / 1000, "voxcpm2", None)
            self._flush_usage()
            log.info("session %s %s%s %s reason=%s pauses=%d audio_ms=%d timings=%s", self.id, resp.id,
                     " (draft)" if resp.draft_text is not None else "", resp.status,
                     resp.cancel_reason, resp.pauses, resp.out.sent_ms if resp.out else 0, resp.timings)
            if resp.started and not resp.pending:  # a discarded draft leaves no trace on the client
                await self._safe_send(
                    {"type": "response.done", "response_id": resp.id, "status": resp.status, "text": said,
                     "reason": resp.cancel_reason, "timings": resp.timings}
                )

    def _record_llm(self, usage: LlmUsage) -> None:
        self._record("llm_input_tokens", usage.input_tokens, self._llm_provider, usage.model)
        self._record("llm_output_tokens", usage.output_tokens, self._llm_provider, usage.model)

    async def _pipeline(self, resp: Response, deltas: AsyncIterator[str]) -> None:
        """LLM text streams in while earlier sentences are already being rendered and played."""
        sentences: asyncio.Queue[str | None] = asyncio.Queue()

        clips = _OPENER_CACHE.get((self.voice["id"], self.cfg.output_sample_rate)) if self._openers_ready() else None
        longest = max((len(o) for o in OPENERS), default=0)

        async def produce() -> None:
            splitter = SentenceSplitter()
            head = clips is not None  # still deciding whether the reply opens with a cached opener
            pending = ""
            async for delta in deltas:
                resp.mark("first_text")
                resp.text += delta
                await self._emit(resp, {"type": "response.text.delta", "response_id": resp.id, "delta": delta})
                if head:
                    pending += delta
                    lead = pending.lstrip()
                    hit = next((o for o in OPENERS if lead.startswith(o)), None)
                    if hit is None and len(lead) < longest and any(o.startswith(lead) for o in OPENERS):
                        continue  # could still become an opener
                    head = False
                    if hit is not None:
                        self._play_cached(resp, hit, clips[hit])
                        delta = lead[len(hit):]
                    else:
                        delta = pending
                for s in splitter.feed(delta):
                    sentences.put_nowait(s)
            if head and pending:
                for s in splitter.feed(pending):
                    sentences.put_nowait(s)
            for s in splitter.flush():
                sentences.put_nowait(s)
            sentences.put_nowait(None)

        async def consume() -> None:
            while (s := await sentences.get()) is not None:
                if self.voice is not None:
                    await self._render_sentence(resp, s)

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(produce())
                tg.create_task(consume())
        except ExceptionGroup as eg:
            raise eg.exceptions[0] from None

    def _play_cached(self, resp: Response, text: str, pcm: bytes) -> None:
        sr = self.cfg.output_sample_rate
        resp.mark("opener")
        resp.sentences.append((resp.audio_ms_queued, text))
        resp.out.push(pcm)
        resp.audio_ms_queued += len(pcm) / (2 * sr) * 1000

    async def _render_sentence(self, resp: Response, sentence: str) -> None:
        voice_id = self.voice["id"]
        sr = self.cfg.output_sample_rate
        resp.sentences.append((resp.audio_ms_queued, sentence))
        resp.tts_chars += len(sentence)
        for attempt in range(2):
            result = TtsResult()
            try:
                async with contextlib.aclosing(self.deps.worker.tts(voice_id, sentence, sr, result)) as chunks:
                    async for chunk in chunks:
                        resp.mark("first_tts_chunk")
                        resp.out.push(chunk)
                        resp.audio_ms_queued += len(chunk) / (2 * sr) * 1000
                resp.tts_audio_ms += result.stats.get("audio_ms", 0)
                return
            except WorkerError as e:
                # A fresh or restarted worker hasn't seen this voice yet.
                if e.code == "voice_not_found" and attempt == 0:
                    await self._push_voice(voice_id)
                    continue
                raise
