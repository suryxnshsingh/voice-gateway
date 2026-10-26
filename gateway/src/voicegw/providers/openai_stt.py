"""OpenAI real-time transcription (gpt-live-transcribe, gpt-realtime-whisper) over WebSocket.

These models stream words while the user speaks but have no turn detection of their own, so the
session marks the end of each turn: on a VAD stop it calls finalize(), which commits the audio
buffer, and the model answers with the final text for that stretch (~0.7 s later in testing).
Input must be 24 kHz PCM16, so the 16 kHz stream is resampled on the way in.
"""

import asyncio
import base64
import contextlib
import json
from collections import OrderedDict, deque
from collections.abc import AsyncIterator

import numpy as np
import soxr
import websockets

from .soniox import SttError, SttUpdate, Token

URL = "wss://api.openai.com/v1/realtime?intent=transcription"
# Models that take a list of candidate languages; the others take at most one.
_MULTI_LANGUAGE = {"gpt-live-transcribe", "gpt-transcribe"}
_MIN_COMMIT_MS = 120


class OpenAIRealtimeSTT:
    provider = "openai"
    needs_endpointing = True

    def __init__(self, ws, model: str):
        self._ws = ws
        self.model = model
        self.audio_bytes = 0  # 16 kHz input bytes, the session's timeline
        self.offset_ms = 0
        self._resampler = soxr.ResampleStream(16000, 24000, 1, dtype="int16")
        self._committed_ms = 0
        self._pending_commits: deque[tuple[int, int]] = deque()
        self._spans: dict[str, tuple[int, int]] = {}
        self._partials: OrderedDict[str, str] = OrderedDict()
        self._closing = False

    @classmethod
    async def connect(cls, *, api_key: str, model: str = "gpt-live-transcribe", language_hints: list[str] | None = None,
                      url: str = URL, delay: str | None = None, keywords: list[str] | None = None) -> "OpenAIRealtimeSTT":
        ws = await websockets.connect(url, additional_headers={"Authorization": f"Bearer {api_key}"},
                                      max_size=2**22, ping_interval=20, ping_timeout=20)
        transcription: dict = {"model": model}
        if delay:
            transcription["delay"] = delay  # "minimal": partial words ~0.5 s sooner (measured Oct 2026)
        if keywords and model in _MULTI_LANGUAGE:
            transcription["keywords"] = keywords
        if language_hints:
            if model in _MULTI_LANGUAGE:
                transcription["languages"] = language_hints
            elif len(language_hints) == 1:
                transcription["language"] = language_hints[0]
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "transcription", "audio": {"input": {
            "format": {"type": "audio/pcm", "rate": 24000},
            "transcription": transcription,
            "turn_detection": None,
            "noise_reduction": None,  # our own RNNoise runs upstream when wanted
        }}}}))
        # Fail fast on a bad key or model instead of on the first word.
        try:
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), 10))
                if event["type"] == "session.updated":
                    break
                if event["type"] == "error":
                    raise SttError(event["error"].get("message", "openai stt error"), status=_status(event["error"]))
        except BaseException:
            await ws.close()
            raise
        return cls(ws, model)

    @property
    def sent_ms(self) -> int:
        return self.audio_bytes // 32

    async def send_audio(self, pcm16k: bytes) -> None:
        self.audio_bytes += len(pcm16k)
        pcm24 = self._resampler.resample_chunk(np.frombuffer(pcm16k, dtype="<i2"))
        if len(pcm24):
            await self._ws.send(json.dumps({"type": "input_audio_buffer.append",
                                            "audio": base64.b64encode(pcm24.astype("<i2").tobytes()).decode()}))

    async def finalize(self) -> None:
        """End of a user turn: commit what's buffered so the model emits its final text."""
        end = self.sent_ms
        if end - self._committed_ms < _MIN_COMMIT_MS:
            return
        self._pending_commits.append((self._committed_ms, end))
        self._committed_ms = end
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    async def end_audio(self) -> None:
        await self.finalize()

    def _partial(self) -> str:
        return "".join(self._partials.values()).strip()

    async def updates(self) -> AsyncIterator[SttUpdate]:
        try:
            async for raw in self._ws:
                e = json.loads(raw)
                kind = e["type"]
                if kind == "input_audio_buffer.committed":
                    if self._pending_commits:
                        self._spans[e["item_id"]] = self._pending_commits.popleft()
                elif kind == "conversation.item.input_audio_transcription.delta":
                    item = e["item_id"]
                    self._partials[item] = self._partials.get(item, "") + (e.get("delta") or "")
                    yield SttUpdate(partial=self._partial())
                elif kind == "conversation.item.input_audio_transcription.completed":
                    item = e["item_id"]
                    self._partials.pop(item, None)
                    start, end = self._spans.pop(item, (self._committed_ms, self.sent_ms))
                    text = (e.get("transcript") or "").strip()
                    # Leading space: one turn can span several committed items, and finals are concatenated.
                    finals = [Token(" " + text, start, end, True)] if text else []
                    yield SttUpdate(finals=finals, partial=self._partial(), endpoint=True)
                elif kind == "conversation.item.input_audio_transcription.failed":
                    self._partials.pop(e.get("item_id"), None)
                elif kind == "error":
                    err = e["error"]
                    if err.get("code") == "input_audio_buffer_commit_empty":
                        continue
                    raise SttError(err.get("message", "openai stt error"), status=_status(err))
        except websockets.ConnectionClosed as e:
            if self._closing:
                return
            raise SttError(f"openai stt connection lost: {e}") from e
        if not self._closing:
            raise SttError("openai closed the stream")

    async def close(self) -> None:
        self._closing = True
        with contextlib.suppress(Exception):
            await self._ws.close()


def _status(err: dict) -> int | None:
    code = str(err.get("code") or "")
    if "invalid_api_key" in code or err.get("type") == "authentication_error":
        return 401
    if "insufficient_quota" in code:
        return 402
    if "rate_limit" in code:
        return 429
    return None


async def transcribe(pcm16k: bytes, *, api_key: str, model: str = "gpt-live-transcribe",
                     language_hints: list[str] | None = None, **connect_kwargs) -> str:
    """Whole-clip transcription (consent recordings): stream it, commit once, wait for the text."""
    stream = await OpenAIRealtimeSTT.connect(api_key=api_key, model=model, language_hints=language_hints, **connect_kwargs)
    try:
        step = 16000 * 2 // 5
        for i in range(0, len(pcm16k), step):
            await stream.send_audio(pcm16k[i : i + step])
        await stream.finalize()
        async with asyncio.timeout(30):
            async for up in stream.updates():
                if up.endpoint:
                    return " ".join(t.text for t in up.finals).strip()
        return ""
    finally:
        await stream.close()
