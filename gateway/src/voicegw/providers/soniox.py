"""Soniox real-time STT over WebSocket.

Soniox re-sends the current non-final hypothesis with every message and sends each final token
once. With endpoint detection on, a final "<end>" token marks the end of a user turn; a
{"type": "finalize"} request produces "<fin>" the same way.
"""

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import websockets

log = logging.getLogger(__name__)

MARKERS = {"<end>", "<fin>"}


@dataclass
class Token:
    text: str
    start_ms: int | None
    end_ms: int | None
    is_final: bool


@dataclass
class SttUpdate:
    finals: list[Token] = field(default_factory=list)
    partial: str = ""
    endpoint: bool = False


class SttError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status  # HTTP-style: 401 bad key, 402 out of credit, 429 rate limited


class SonioxStream:
    provider = "soniox"
    needs_endpointing = False  # Soniox detects the end of a turn itself ("<end>")

    def __init__(self, ws, model: str, keepalive_s: float = 8.0):
        self._ws = ws
        self.model = model
        self.offset_ms = 0  # where this stream's t=0 sits on the session's audio timeline
        self._keepalive_s = keepalive_s
        self._last_send = asyncio.get_running_loop().time()
        self._ka_task = asyncio.create_task(self._keepalive())
        self.audio_bytes = 0
        self._closing = False

    @classmethod
    async def connect(cls, *, api_key: str, url: str, model: str, sample_rate: int = 16000,
                      language_hints: list[str] | None = None, endpoint_detection: bool = True) -> "SonioxStream":
        ws = await websockets.connect(url, ping_interval=20, ping_timeout=20, max_size=2**22)
        config = {
            "api_key": api_key,
            "model": model,
            "audio_format": "pcm_s16le",
            "sample_rate": sample_rate,
            "num_channels": 1,
            "enable_endpoint_detection": endpoint_detection,
        }
        if language_hints:
            config["language_hints"] = language_hints
        await ws.send(json.dumps(config))
        return cls(ws, model)

    async def _keepalive(self) -> None:
        # Soniox drops a stream that goes quiet; a muted client still needs to hold its session.
        with contextlib.suppress(Exception):
            while True:
                await asyncio.sleep(self._keepalive_s / 2)
                loop = asyncio.get_running_loop()
                if loop.time() - self._last_send >= self._keepalive_s:
                    await self._ws.send(json.dumps({"type": "keepalive"}))
                    self._last_send = loop.time()

    async def send_audio(self, pcm: bytes) -> None:
        await self._ws.send(pcm)
        self.audio_bytes += len(pcm)
        self._last_send = asyncio.get_running_loop().time()

    async def finalize(self) -> None:
        await self._ws.send(json.dumps({"type": "finalize"}))

    async def end_audio(self) -> None:
        await self._ws.send("")

    async def updates(self) -> AsyncIterator[SttUpdate]:
        try:
            async for raw in self._ws:
                data = json.loads(raw)
                if data.get("error_code"):
                    code = data.get("error_code")
                    raise SttError(f"{code}: {data.get('error_message')}", status=int(code) if str(code).isdigit() else None)
                tokens = data.get("tokens") or []
                if tokens:
                    up = SttUpdate()
                    partial = []
                    for t in tokens:
                        tok = Token(t.get("text", ""), t.get("start_ms"), t.get("end_ms"), bool(t.get("is_final")))
                        if tok.text in MARKERS:
                            up.endpoint = up.endpoint or tok.is_final
                        elif tok.is_final:
                            up.finals.append(tok)
                        else:
                            partial.append(tok.text)
                    up.partial = "".join(partial)
                    yield up
                if data.get("finished"):
                    return
        except websockets.ConnectionClosed as e:
            if self._closing:
                return
            raise SttError(f"soniox connection lost: {e}") from e
        # The iterator also ends quietly on a clean close. Unless we closed it, that's a dead stream.
        if not self._closing:
            raise SttError("soniox closed the stream")

    async def close(self) -> None:
        self._closing = True
        self._ka_task.cancel()
        with contextlib.suppress(Exception):
            await self._ws.close()


async def transcribe(pcm16k: bytes, **connect_kwargs) -> str:
    """Transcribe a short clip (e.g. a consent recording) by streaming it and waiting for the end."""
    stream = await SonioxStream.connect(endpoint_detection=False, **connect_kwargs)
    try:
        async def feed():
            step = 16000 * 2 // 5  # 200 ms
            for i in range(0, len(pcm16k), step):
                await stream.send_audio(pcm16k[i : i + step])
            await stream.end_audio()

        feeder = asyncio.create_task(feed())
        text = []
        async for up in stream.updates():
            text.extend(t.text for t in up.finals)
        await feeder
        return "".join(text).strip()
    finally:
        await stream.close()
