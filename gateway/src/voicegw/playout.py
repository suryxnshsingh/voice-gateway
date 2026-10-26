"""Server-side buffer for the assistant's audio.

TTS renders faster than real time, but the audio is sent at playback speed, keeping only `lead_ms`
queued on the client. That keeps the server in control of playback: it can pause, resume or drop
the rest of a reply instantly, and it knows how far the listener got (what was actually heard).
"""

import asyncio
import contextlib
import time
from collections.abc import Awaitable, Callable


class Playout:
    def __init__(self, send: Callable[[bytes], Awaitable[None]], sample_rate: int, lead_ms: int = 200,
                 frame_ms: int = 20, on_first_frame: Callable[[], Awaitable[None]] | None = None,
                 prebuffer_ms: int = 0):
        self._send = send
        self._bpms = sample_rate * 2 / 1000  # bytes per ms
        self._frame = int(self._bpms * frame_ms) // 2 * 2
        self._lead = lead_ms
        self._on_first = on_first_frame
        self._pending = bytearray()
        self._wake = asyncio.Event()
        self._ended = False
        self.sent_ms = 0.0
        # Playback clock: how much the client has played. It runs only while audio is flowing, so
        # pauses and underruns (TTS slower than real time) don't push the estimate ahead.
        self._base_ms = 0.0
        self._anchor: float | None = None
        self.paused = False
        self.drained = asyncio.Event()
        # Hold the first frame until this much audio is rendered. For TTS that runs near real time
        # (VoxCPM2 on a Mac), so playback doesn't catch up with generation and stutter.
        self._prebuffer = prebuffer_ms
        self.queued_ms = 0.0
        # The client ran dry mid-reply (TTS slower than real time): count and total silence heard.
        self.underruns = 0
        self.underrun_ms = 0.0
        self._task = asyncio.create_task(self._run())

    def played_ms(self) -> float:
        if self._anchor is None:
            return self._base_ms
        return min(self.sent_ms, self._base_ms + (time.monotonic() - self._anchor) * 1000)

    def push(self, pcm: bytes) -> None:
        self._pending += pcm
        self.queued_ms += len(pcm) / self._bpms
        self._wake.set()

    def end(self) -> None:
        """No more audio is coming; `drained` fires once the client has played what was sent."""
        self._ended = True
        self._wake.set()

    def pause(self) -> None:
        if not self.paused:
            self._base_ms = self.played_ms()
            self._anchor = None
            self.paused = True

    def resume(self) -> None:
        if self.paused:
            self.paused = False
            # The client resumes with its buffered lead, so the clock restarts from where it froze.
            self._anchor = time.monotonic()
            self._wake.set()

    async def close(self) -> None:
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

    async def _run(self) -> None:
        while True:
            if self.paused:
                await self._wait()
                continue
            if not self._pending:
                if self._ended:
                    remaining = self.sent_ms - self.played_ms()
                    if remaining <= 0:
                        self.drained.set()
                        return
                    await self._wait(remaining / 1000)
                    continue
                await self._wait()
                continue
            if self.sent_ms == 0 and not self._ended and self.queued_ms < self._prebuffer:
                await self._wait()
                continue
            ahead = self.sent_ms - self.played_ms()
            if ahead >= self._lead:
                await self._wait((ahead - self._lead) / 1000 + 0.002)
                continue
            if self._anchor is not None and ahead <= 0 and self.sent_ms > 0:
                self.underruns += 1
                self.underrun_ms += self._base_ms + (time.monotonic() - self._anchor) * 1000 - self.sent_ms
            if self._anchor is None or ahead <= 0:
                # Starting, resuming, or recovering from an underrun: restart the clock from here.
                self._base_ms = self.sent_ms if ahead <= 0 else self._base_ms
                self._anchor = time.monotonic()
                if self.sent_ms == 0 and self._on_first is not None:
                    await self._on_first()
            frame = bytes(self._pending[: self._frame])
            del self._pending[: self._frame]
            await self._send(frame)
            self.sent_ms += len(frame) / self._bpms

    async def _wait(self, timeout: float | None = None) -> None:
        self._wake.clear()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._wake.wait(), timeout)
