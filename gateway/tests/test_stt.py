import asyncio
import json

import pytest

from voicegw.providers import stt as stt_chain
from voicegw.providers.openai_stt import OpenAIRealtimeSTT
from voicegw.providers.soniox import SttError
from voicegw.session import Turn

from test_session import FakeWS, make_session  # noqa: F401  (fixture)


class ScriptedWS:
    def __init__(self, events):
        self.sent, self._events = [], [json.dumps(e) for e in events]

    async def send(self, m):
        self.sent.append(json.loads(m))

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self._events:
            yield e

    async def close(self):
        pass


async def test_openai_events_map_to_partials_finals_and_endpoint():
    ws = ScriptedWS([
        {"type": "conversation.item.input_audio_transcription.delta", "item_id": "a", "delta": " What is"},
        {"type": "conversation.item.input_audio_transcription.delta", "item_id": "a", "delta": " DNA"},
        {"type": "input_audio_buffer.committed", "item_id": "a"},
        {"type": "conversation.item.input_audio_transcription.completed", "item_id": "a", "transcript": "What is DNA?"},
        {"type": "error", "error": {"code": "input_audio_buffer_commit_empty", "message": "empty"}},
    ])
    s = OpenAIRealtimeSTT(ws, "gpt-live-transcribe")
    await s.send_audio(b"\x00\x00" * 16000)  # 1 s
    await s.finalize()
    await s.finalize()  # nothing new buffered: no second commit
    assert [m["type"] for m in ws.sent].count("input_audio_buffer.commit") == 1
    ups = []
    with pytest.raises(SttError):  # the scripted socket then "closes" on us, which is a failure
        async for u in s.updates():
            ups.append(u)
    assert [u.partial for u in ups[:2]] == ["What is", "What is DNA"]
    final = ups[2]
    assert final.endpoint and final.finals[0].text.strip() == "What is DNA?"
    assert (final.finals[0].start_ms, final.finals[0].end_ms) == (0, 1000)


async def test_auth_error_maps_to_status():
    ws = ScriptedWS([{"type": "error", "error": {"code": "insufficient_quota", "message": "no credit"}}])
    with pytest.raises(SttError) as e:
        async for _ in OpenAIRealtimeSTT(ws, "m").updates():
            pass
    assert e.value.status == 402


class FakeStream:
    needs_endpointing = False

    def __init__(self, provider):
        self.provider, self.model, self.offset_ms, self.audio_bytes = provider, "m", 0, 0
        self.received = bytearray()
        self.fail = asyncio.Event()

    async def send_audio(self, pcm):
        self.audio_bytes += len(pcm)
        self.received += pcm

    async def finalize(self):
        pass

    async def updates(self):
        await self.fail.wait()
        raise SttError("402: balance exhausted", status=402)
        yield  # pragma: no cover

    async def close(self):
        pass


async def test_failover_replays_current_turn_to_next_provider(make_session, monkeypatch):
    s, ws, *_ = make_session()
    streams = []

    async def fake_connect(settings, hints):
        st = FakeStream(["soniox", "openai"][len(streams)])
        streams.append(st)
        return st

    monkeypatch.setattr(stt_chain, "connect", fake_connect)
    monkeypatch.setattr(stt_chain, "mark_down", lambda p, e: None)
    s.audio = type(s.audio)()  # empty timeline
    for _ in range(50):  # 2 s of audio in 40 ms frames
        await s._on_audio(b"\x01\x00" * 640)
    s._turn = Turn(final="what is", start_ms=1500, end_ms=1900)
    streams[0].fail.set()
    for _ in range(100):
        if len(streams) == 2:
            break
        await asyncio.sleep(0.01)
    new = streams[1]
    assert new.offset_ms == 1200  # turn start minus 300 ms of lead-in
    assert len(new.received) == (2000 - 1200) * 32  # the turn's audio was re-sent
    assert s._turn.final == ""  # the new provider rebuilds the turn from the replay
    assert [e["provider"] for e in ws.of("stt.connected")] == ["soniox", "openai"]
    assert not ws.of("error")
