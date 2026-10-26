import asyncio
import io
import wave

SAMPLE_RATE_IN = 16000
BYTES_PER_MS_IN = SAMPLE_RATE_IN * 2 // 1000


class AudioError(Exception):
    pass


async def ffmpeg_convert(data: bytes, sample_rate: int, fmt: str = "wav") -> bytes:
    """Any container ffmpeg can read (m4a, mp3, webm/opus, wav) -> mono at sample_rate.

    fmt "wav" gives a PCM16 WAV file; "s16le" gives raw PCM16.
    """
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-ac", "1", "-ar", str(sample_rate), "-f", "s16le", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(data)
    if proc.returncode != 0 or not out:
        raise AudioError(err.decode(errors="replace").strip() or "could not decode audio")
    # ffmpeg can't seek back on a pipe to fill in a WAV header's sizes, so write the header here.
    return wav_bytes(out, sample_rate) if fmt == "wav" else out


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def pcm_seconds(pcm: bytes, sample_rate: int) -> float:
    return len(pcm) / (2 * sample_rate)


class PcmRing:
    """Recent input audio, addressable by stream time in ms (the timeline Soniox timestamps use)."""

    def __init__(self, seconds: int = 120):
        self._buf = bytearray()
        self._base = 0  # stream byte offset of _buf[0]
        self._max = seconds * 1000 * BYTES_PER_MS_IN

    def append(self, pcm: bytes) -> None:
        self._buf += pcm
        overflow = len(self._buf) - self._max
        if overflow > 0:
            overflow += overflow % 2
            del self._buf[:overflow]
            self._base += overflow

    @property
    def end_ms(self) -> int:
        return (self._base + len(self._buf)) // BYTES_PER_MS_IN

    def slice(self, start_ms: int, end_ms: int) -> bytes:
        lo = max(start_ms * BYTES_PER_MS_IN - self._base, 0)
        hi = max(end_ms * BYTES_PER_MS_IN - self._base, 0)
        return bytes(self._buf[lo:hi])


# Public output formats: (ffmpeg args, content type).
FORMATS = {
    "mp3": (["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"], "audio/mpeg"),
    "opus": (["-c:a", "libopus", "-b:a", "32k", "-f", "ogg"], "audio/ogg"),
    "wav": (["-c:a", "pcm_s16le", "-f", "wav"], "audio/wav"),
    "pcm": (None, "audio/pcm"),  # raw PCM16 LE mono at the requested sample rate
}


async def encode(pcm: bytes, sample_rate: int, fmt: str) -> bytes:
    """Whole PCM16 mono buffer -> a complete file in `fmt`."""
    if fmt == "pcm":
        return pcm
    if fmt == "wav":
        return wav_bytes(pcm, sample_rate)
    args, _ = FORMATS[fmt]
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
        "-i", "pipe:0", *args, "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(pcm)
    if proc.returncode != 0:
        raise AudioError(err.decode(errors="replace").strip() or "encoding failed")
    return out


class StreamEncoder:
    """PCM16 in, encoded bytes out, as it goes (an ffmpeg pipe), for streaming mp3/opus responses."""

    def __init__(self, sample_rate: int, fmt: str):
        self._sr, self._fmt = sample_rate, fmt
        self._proc: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        args, _ = FORMATS[self._fmt]
        self._proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", str(self._sr), "-ac", "1",
            "-i", "pipe:0", *args, "-flush_packets", "1", "pipe:1",
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )

    async def write(self, pcm: bytes) -> None:
        self._proc.stdin.write(pcm)
        await self._proc.stdin.drain()

    async def finish(self) -> None:
        self._proc.stdin.close()

    async def read(self) -> bytes:
        return await self._proc.stdout.read(4096)

    def kill(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            self._proc.kill()
