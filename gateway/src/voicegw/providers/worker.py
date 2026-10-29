"""Client for the internal model worker (VoxCPM2 TTS, emotion2vec tone, ECAPA speaker similarity)."""

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
import websockets


FIRST_CHUNK_TIMEOUT = 20.0  # includes waiting for a free TTS instance (the worker gives up at 15 s)
CHUNK_TIMEOUT = 10.0


class WorkerError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass
class TtsResult:
    status: str = "pending"
    stats: dict = field(default_factory=dict)


class WorkerClient:
    def __init__(self, base_url: str, token: str = ""):
        self._base = base_url.rstrip("/")
        self._ws_base = self._base.replace("http://", "ws://").replace("https://", "wss://")
        self._headers = {"x-worker-token": token} if token else {}
        self._http = httpx.AsyncClient(base_url=self._base, headers=self._headers, timeout=60)

    async def close(self) -> None:
        await self._http.aclose()

    async def health(self) -> dict:
        r = await self._http.get("/health", timeout=5)
        r.raise_for_status()
        return r.json()

    async def put_voice(self, voice_id: str, wav: bytes) -> dict:
        r = await self._http.put(f"/voices/{voice_id}", content=wav, timeout=120)
        r.raise_for_status()
        return r.json()

    async def has_voice(self, voice_id: str) -> bool:
        r = await self._http.get(f"/voices/{voice_id}", timeout=5)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        return True

    async def delete_voice(self, voice_id: str) -> None:
        r = await self._http.delete(f"/voices/{voice_id}")
        r.raise_for_status()

    async def tone(self, pcm16k: bytes) -> dict:
        r = await self._http.post("/tone", content=pcm16k, timeout=10)
        r.raise_for_status()
        return r.json()

    async def speaker_similarity(self, a_wav: bytes, b_wav: bytes) -> float:
        r = await self._http.post("/speaker/similarity", files={"a": ("a.wav", a_wav), "b": ("b.wav", b_wav)})
        r.raise_for_status()
        return r.json()["score"]

    async def tts(self, voice_id: str, text: str, sample_rate: int, result: TtsResult,
                  timesteps: int | None = None, adapter: str | None = None) -> AsyncIterator[bytes]:
        """Stream PCM16 for one sentence. Closing the iterator early cancels generation on the worker.
        `adapter` is the LoRA adapter of a fine-tuned voice; without it the voice renders on the base model."""
        async with websockets.connect(f"{self._ws_base}/tts", additional_headers=self._headers, max_size=2**22) as ws:
            req = {"voice_id": voice_id, "text": text, "sample_rate": sample_rate}
            if timesteps:
                req["inference_timesteps"] = timesteps
            if adapter:
                req["adapter"] = adapter
            await ws.send(json.dumps(req))
            try:
                first = True
                while True:
                    # A stalled worker must turn into a visible error, not a reply that never speaks.
                    try:
                        msg = await asyncio.wait_for(ws.recv(), FIRST_CHUNK_TIMEOUT if first else CHUNK_TIMEOUT)
                    except asyncio.TimeoutError:
                        raise WorkerError("tts_timeout") from None
                    first = False
                    if isinstance(msg, bytes):
                        yield msg
                        continue
                    data = json.loads(msg)
                    if data["type"] == "error":
                        raise WorkerError(data["code"])
                    result.status, result.stats = data["type"], data
                    return
            finally:
                if result.status == "pending":
                    result.status = "cancelled"
                    with contextlib.suppress(Exception):
                        await ws.send(json.dumps({"type": "cancel"}))
