"""Regressions for bugs found in review (2026-10-26). Each test is the scenario that broke."""

import asyncio

import numpy as np

from voicegw import session as sess
from voicegw.frontend import FrontendResult, VadEvent
from voicegw.providers.soniox import SttError, SttUpdate

from test_session import FakeWorker, FakeWS, _CountingStt, _FixedTurn, make_session, utterance, wait_for  # noqa: F401


class _SilentFrontend:
    params = None

    def process(self, pcm):
        return FrontendResult(denoised=np.zeros(0, np.float32))


class _Stream:
    provider, model, needs_endpointing = "fake", "m", False

    def __init__(self, fail=None, end_quietly=False):
        self.offset_ms = self.audio_bytes = 0
        self.fail, self.end_quietly = fail, end_quietly
        self.closed, self.dead, self.sent = False, False, []

    async def send_audio(self, pcm):
        if self.dead:
            raise ConnectionError("send on a closed socket")
        self.sent.append(len(pcm))
        self.audio_bytes += len(pcm)

    async def finalize(self):
        pass

    async def close(self):
        self.closed = True

    async def updates(self):
        await asyncio.sleep(0.05)
        if self.fail:
            raise self.fail
        if self.end_quietly:
            self.dead = True
            return
        await asyncio.sleep(3600)
        yield SttUpdate()


async def test_quietly_closed_stt_fails_over_and_mic_frames_survive(make_session, monkeypatch):
    s, ws, *_ = make_session()
    s.frontend = _SilentFrontend()
    made = []

    async def connect(settings, hints):
        made.append(_Stream())
        return made[-1]

    monkeypatch.setattr(sess.stt_chain, "connect", connect)
    monkeypatch.setattr(sess.stt_chain, "mark_down", lambda *a: None)
    dead = _Stream(end_quietly=True)
    s.stt = dead
    s._spawn(s._read_stt(dead))
    await asyncio.sleep(0.1)
    await s._on_audio(b"\x00\x00" * 320)  # must not raise on the dead socket
    assert s.stt is made[0]


async def test_failover_opens_one_stream_and_it_gets_the_replay(make_session, monkeypatch):
    s, ws, *_ = make_session()
    s.frontend = _SilentFrontend()
    made = []

    async def slow_connect(settings, hints):
        await asyncio.sleep(0.2)
        made.append(_Stream())
        return made[-1]

    monkeypatch.setattr(sess.stt_chain, "connect", slow_connect)
    monkeypatch.setattr(sess.stt_chain, "mark_down", lambda *a: None)
    failing = _Stream(fail=SttError("boom"))
    s.stt = failing
    s._turn.start_ms = 2000
    s._spawn(s._read_stt(failing))
    for _ in range(20):  # the mic keeps sending while the new provider connects
        await s._on_audio(b"\x00\x00" * 320)
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.3)
    assert len(made) == 1 and s.stt is made[0] and made[0].sent  # one stream, and it got the replay


async def test_resume_after_smart_turn_commit_still_commits_the_rest(make_session):
    s, ws, *_ = make_session()
    s.stt, s._turn_detector = _CountingStt(), _FixedTurn(0.9)
    s.cfg.turn_silence_ms = 600
    await s._on_stt(SttUpdate(partial="What is DNA"))
    await s._on_vad(VadEvent("start"))
    await s._on_vad(VadEvent("pause", 900))
    await asyncio.sleep(0.05)
    await s._on_vad(VadEvent("resume"))
    await s._on_stt(SttUpdate(partial="What is DNA and RNA"))
    await s._on_vad(VadEvent("stop", 1500))
    await asyncio.sleep(0.4)
    assert s.stt.finalized == 2 and not s._turn_finalized


class _SlowWS(FakeWS):
    async def send_json(self, e):
        await asyncio.sleep(0.005)
        self.events.append(e)


async def test_speech_during_draft_release_still_finishes_the_reply(make_session, tmp_path):
    s, ws, *_ = make_session()
    s.ws = slow = _SlowWS()
    await s._on_stt(SttUpdate(partial="What is photosynthesis"))
    await s._on_vad(VadEvent("start"))
    await s._on_vad(VadEvent("stop", 800))
    await asyncio.sleep(0.15)
    release = asyncio.create_task(s._on_stt(utterance("What is photosynthesis?")))
    await asyncio.sleep(0.03)  # mid-drain
    await s._on_vad(VadEvent("start"))
    await release
    await asyncio.sleep(0.2)
    assert len(slow.of("response.started")) == 1 and len(slow.of("response.done")) == 1


async def test_second_cough_restarts_the_resume_timer(make_session):
    s, ws, *_ = make_session(worker=FakeWorker(chunk_delay=0.005, chunks=60))
    s.cfg.false_interruption_ms = 300
    await s._on_stt(utterance("Explain photosynthesis"))
    await wait_for(ws, "response.audio.started")
    await asyncio.sleep(0.15)
    loop = asyncio.get_running_loop()
    await s._on_vad(VadEvent("start"))
    await s._on_vad(VadEvent("stop", 250))
    await asyncio.sleep(0.15)
    await s._on_vad(VadEvent("start"))
    await asyncio.sleep(0.12)
    await s._on_vad(VadEvent("stop", 100))
    t_stop = loop.time()
    await wait_for(ws, "output_audio.resumed")
    assert loop.time() - t_stop >= 0.29


async def test_echo_of_a_finished_reply_is_not_a_turn(make_session):
    s, ws, *_ = make_session()
    await s._on_stt(utterance("What is photosynthesis?"))
    await wait_for(ws, "response.done")
    await s._on_stt(utterance("That is photosynthesis.", 5000, 5600))  # the reply's last words, via the mic
    await asyncio.sleep(0.3)
    assert [e.get("ignored") for e in ws.of("transcript.done")] == [None, "echo"]
    assert len(s.deps.llm.calls) == 1
