import asyncio
import time

from voicegw.playout import Playout


class Sink:
    def __init__(self):
        self.frames, self.times = [], []

    async def __call__(self, b):
        self.frames.append(b)
        self.times.append(time.monotonic())


async def test_paces_at_real_time_with_bounded_lead():
    sink = Sink()
    p = Playout(sink, 16000, lead_ms=100, frame_ms=20)
    p.push(b"\x00" * 32 * 1000)  # 1 s of audio, all at once
    p.end()
    t0 = time.monotonic()
    await asyncio.wait_for(p.drained.wait(), 3)
    elapsed = time.monotonic() - t0
    assert 0.9 <= elapsed <= 1.3
    # never more than the lead (plus one frame) ahead of the playback clock
    first = sink.times[0]
    for i, t in enumerate(sink.times):
        assert (i + 1) * 20 - (t - first) * 1000 <= 100 + 20 + 15


async def test_pause_freezes_clock_and_sending():
    sink = Sink()
    p = Playout(sink, 16000, lead_ms=60, frame_ms=20)
    p.push(b"\x00" * 32 * 2000)
    await asyncio.sleep(0.3)
    p.pause()
    played, sent = p.played_ms(), len(sink.frames)
    await asyncio.sleep(0.2)
    assert p.played_ms() == played and len(sink.frames) == sent
    p.resume()
    await asyncio.sleep(0.2)
    assert p.played_ms() > played + 150 and len(sink.frames) > sent
    await p.close()


async def test_underrun_does_not_run_clock_ahead():
    sink = Sink()
    p = Playout(sink, 16000, lead_ms=60, frame_ms=20)
    p.push(b"\x00" * 32 * 100)  # 100 ms, then a gap while "TTS" is slow
    await asyncio.sleep(0.4)
    assert p.played_ms() <= p.sent_ms == 100
    p.push(b"\x00" * 32 * 100)
    await asyncio.sleep(0.05)
    assert p.played_ms() < 160  # restarted from 100, not from wall-clock time
    await p.close()


async def test_prebuffer_holds_first_frame_and_underruns_are_counted():
    sink = Sink()
    p = Playout(sink, 16000, lead_ms=60, frame_ms=20, prebuffer_ms=100)
    p.push(b"\x00" * 32 * 60)
    await asyncio.sleep(0.1)
    assert sink.frames == []  # 60 ms rendered < 100 ms prebuffer
    p.push(b"\x00" * 32 * 60)
    await asyncio.sleep(0.3)  # plays 120 ms, then runs dry
    p.push(b"\x00" * 32 * 40)
    await asyncio.sleep(0.1)
    assert sink.frames and p.underruns == 1 and p.underrun_ms > 100
    await p.close()
