"""Scripted client for the realtime endpoint: plays audio files in as user turns at real-time pace."""

import asyncio
import contextlib
import json
import time
import wave

import websockets

from .audio import ffmpeg_convert

FRAME_MS = 40


def trailing_silence_s(pcm: bytes, threshold: int = 300) -> float:
    """Seconds of near-silence at the end of a clip, so latency counts from the last word."""
    import array
    a = array.array("h", pcm)
    i = len(a) - 1
    while i > 0 and abs(a[i]) < threshold:
        i -= 1
    return (len(a) - 1 - i) / 16000


async def talk(url: str, api_key: str, voice: str, files: list[str], out: str, instructions: str,
               draft: bool = True) -> None:
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {api_key}"}, max_size=2**22) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {"voice_id": voice, "instructions": instructions,
                                                                         "preemptive": draft}}))
        done = asyncio.Event()
        audio = bytearray()
        sample_rate = 24000
        reply_no = 0
        t_turn = 0.0
        t_speech_end = 0.0

        async def receive():
            nonlocal sample_rate, reply_no, t_turn, t_speech_end
            async for msg in ws:
                if isinstance(msg, bytes):
                    audio.extend(msg)
                    continue
                e = json.loads(msg)
                kind = e["type"]
                if kind == "transcript.done":
                    t_turn = time.monotonic()
                    print(f"\nuser: {e['text']}")
                elif kind == "tone":
                    print(f"  tone ({e['source']}): {e['label']} {e.get('confidence', '')} {e.get('reason', '')}")
                elif kind == "response.audio.started":
                    sample_rate = e["sample_rate"]
                    now = time.monotonic()
                    print(f"  first audio {1000 * (now - t_speech_end):.0f} ms after the speech ended "
                          f"({1000 * (now - t_turn):.0f} ms after transcript.done)")
                elif kind == "response.done":
                    print(f"assistant [{e['status']}]: {e['text']}\n  timings: {e['timings']}")
                    reply_no += 1
                    path = f"{out}_{reply_no}.wav"
                    with wave.open(path, "wb") as w:
                        w.setnchannels(1)
                        w.setsampwidth(2)
                        w.setframerate(sample_rate)
                        w.writeframes(bytes(audio))
                    print(f"  saved {path} ({len(audio) / (2 * sample_rate):.1f} s)")
                    audio.clear()
                    done.set()
                elif kind == "error":
                    print(f"  error: {e['code']}: {e['message']}")
                elif kind in ("session.updated",):
                    print(f"session voice={e['session']['voice_id']}")

        receiver = asyncio.create_task(receive())
        step = 16000 * 2 * FRAME_MS // 1000
        silence = bytes(step)
        for path in files:
            pcm = await ffmpeg_convert(open(path, "rb").read(), 16000, "s16le")
            done.clear()
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i : i + step])
                await asyncio.sleep(FRAME_MS / 1000)
            t_speech_end = time.monotonic() - trailing_silence_s(pcm)
            # Keep the mic "open" with silence until the reply finishes, like a real caller.
            t_end = time.monotonic() + 30
            while not done.is_set() and time.monotonic() < t_end:
                await ws.send(silence)
                await asyncio.sleep(FRAME_MS / 1000)
            for _ in range(int(1000 / FRAME_MS)):
                await ws.send(silence)
                await asyncio.sleep(FRAME_MS / 1000)
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver
