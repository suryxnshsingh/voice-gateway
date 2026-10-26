"""Session orchestration with fake STT, LLM and worker: turn -> tone -> reply -> audio, barge-in, carry-over."""

import asyncio
import json

import pytest

from voicegw.config import Settings
from voicegw.db import DB
from voicegw.frontend import VadEvent
from voicegw.providers.soniox import SttUpdate, Token
from voicegw.session import Deps, Session


class FakeWS:
    def __init__(self):
        self.events, self.audio = [], bytearray()
        self.inbox: asyncio.Queue = asyncio.Queue()

    async def send_json(self, e):
        self.events.append(e)

    async def send_bytes(self, b):
        self.audio += b

    async def receive(self):
        return await self.inbox.get()

    async def close(self, code=1000):
        pass

    def types(self):
        return [e["type"] for e in self.events]

    def of(self, kind):
        return [e for e in self.events if e["type"] == kind]


class FakeLLM:
    available = True

    async def warm(self, model):
        pass

    def __init__(self, reply="Sure. Plants use sunlight to make sugar from water and carbon dioxide. That is photosynthesis."):
        self.reply, self.calls = reply, []

    async def stream(self, model, messages, max_tokens, usage, reasoning_effort=None):
        self.calls.append(messages)
        usage.input_tokens, usage.output_tokens = 100, 20
        for i in range(0, len(self.reply), 8):
            await asyncio.sleep(0.005)
            yield self.reply[i : i + 8]

    async def json(self, model, messages, max_tokens, usage):
        return json.dumps({"tone": "confused", "confidence": 0.7, "reason": "asks to repeat"})


class FakeWorker:
    def __init__(self, chunk_delay=0.01, chunks=5):
        self.delay, self.n, self.spoken = chunk_delay, chunks, []

    async def tone(self, pcm):
        return {"label": "neutral", "confidence": 0.9, "scores": {"neutral": 0.9, "sad": 0.1}}

    async def has_voice(self, voice_id):
        return True

    async def health(self):
        return {"tts": {"model": "fake"}}

    async def tts(self, voice_id, text, sr, result):
        self.spoken.append(text)
        for _ in range(self.n):
            await asyncio.sleep(self.delay)
            yield b"\x01\x00" * 480
        result.status, result.stats = "done", {"audio_ms": self.n * 20}


@pytest.fixture
def make_session(tmp_path):
    def make(llm=None, worker=None):
        settings = Settings(data_dir=tmp_path, tone_wait_ms=50, playout_lead_ms=40)
        db = DB(settings.db_path)
        cust = db.create_customer("Test")
        voice = db.create_voice(customer_id=cust.id, name="v", speaker_name="s", reference_seconds=10,
                                consent_phrase=None, consent_code=None, consent_expires_at=None, status="active")
        ws = FakeWS()
        s = Session(ws, cust, Deps(db, worker or FakeWorker(), llm or FakeLLM(), settings))
        s.voice = voice
        s.cfg.voice_id = voice["id"]
        s.audio.append(b"\x00\x00" * 16000 * 3)
        return s, ws, db, cust
    return make


def utterance(text, start=100, end=900, endpoint=True):
    return SttUpdate(finals=[Token(text, start, end, True)], partial="", endpoint=endpoint)


async def wait_for(ws, kind, n=1, timeout=3.0):
    async def poll():
        while len(ws.of(kind)) < n:
            await asyncio.sleep(0.005)
    await asyncio.wait_for(poll(), timeout)


async def test_turn_produces_tone_reply_and_audio(make_session):
    s, ws, db, cust = make_session()
    await s._on_stt(utterance("What is photosynthesis?"))
    await wait_for(ws, "response.done")
    await wait_for(ws, "tone", 2)
    done = ws.of("response.done")[0]
    assert done["status"] == "completed"
    assert done["text"].startswith("Sure.")
    assert {"first_text", "first_audio"} <= done["timings"].keys()
    assert len(ws.audio) > 0
    assert [t["source"] for t in ws.of("tone")] == ["acoustic", "fused"]
    # tone arrived within tone_wait_ms, so the LLM saw it (fused, if fusion beat the LLM call)
    assert s.deps.llm.calls[0][-1]["content"].split("\n")[-1] in ("[tone: neutral]", "[tone: confused]")
    # later turns see the fused label
    assert s.history[0]["tone"] == "confused"
    assert s.history[-1]["role"] == "assistant"
    kinds = {r["kind"] for r in db.usage_summary(cust.id, 0, 1e12)}
    assert {"llm_input_tokens", "tts_characters", "tts_seconds", "tone_calls"} <= kinds


async def test_barge_in_cancels_reply_and_keeps_only_heard_text(make_session):
    s, ws, *_ = make_session(worker=FakeWorker(chunk_delay=0.05, chunks=20))
    await s._on_stt(utterance("Explain photosynthesis"))
    await wait_for(ws, "response.audio.started")
    await asyncio.sleep(0.1)
    await s._on_stt(SttUpdate(partial="wait stop"))
    await wait_for(ws, "response.done")
    done = ws.of("response.done")[0]
    assert done["status"] == "cancelled" and done["reason"] == "barge_in"
    # the reply was cut during its first sentence, so only that sentence counts as said
    assert s.history[-1] == {"role": "assistant", "content": done["text"]}
    assert done["text"] == s.deps.worker.spoken[0]


async def test_talking_before_any_audio_merges_into_one_turn(make_session):
    s, ws, *_ = make_session(llm=FakeLLM(), worker=FakeWorker())
    s.s.tone_wait_ms = 400
    s.deps.worker.tone = _slow_tone
    await s._on_stt(utterance("What is photosynthesis", 100, 900))
    await asyncio.sleep(0.02)  # reply is still waiting on tone; nothing played yet
    await s._on_stt(SttUpdate(partial="and why"))
    await s._on_stt(utterance("and why does it matter?", 1200, 2000))
    await wait_for(ws, "response.done")
    assert [e["text"] for e in ws.of("transcript.done")] == [
        "What is photosynthesis", "What is photosynthesis and why does it matter?",
    ]
    users = [m["content"] for m in s.history if m["role"] == "user"]
    assert users == ["What is photosynthesis and why does it matter?"]


async def _slow_tone(pcm):
    await asyncio.sleep(0.3)
    return {"label": "neutral", "confidence": 0.9, "scores": {"neutral": 0.9}}


async def test_response_create_speaks_given_text_without_llm(make_session):
    s, ws, *_ = make_session()
    s.cfg.llm = False
    await s._on_event({"type": "response.create", "text": "Namaste! Aaj hum DNA padhenge."})
    await wait_for(ws, "response.done")
    assert ws.of("response.done")[0]["text"] == "Namaste! Aaj hum DNA padhenge."
    assert s.deps.llm.calls == []


async def test_inactive_voice_is_rejected(make_session):
    s, ws, db, cust = make_session()
    pending = db.create_voice(customer_id=cust.id, name="p", speaker_name="s", reference_seconds=10,
                              consent_phrase="x", consent_code="a b c", consent_expires_at=0)
    await s._on_event({"type": "session.update", "session": {"voice_id": pending["id"]}})
    assert ws.of("error")[0]["code"] == "voice_unavailable"
    assert s.cfg.voice_id != pending["id"]


async def _playing(make_session, **cfg):
    s, ws, *_ = make_session(worker=FakeWorker(chunk_delay=0.005, chunks=60))  # 1.2 s of audio per sentence
    for k, v in cfg.items():
        setattr(s.cfg, k, v)
    await s._on_stt(utterance("Explain photosynthesis"))
    await wait_for(ws, "response.audio.started")
    await asyncio.sleep(0.15)
    return s, ws


async def test_vad_pauses_then_words_cancel(make_session):
    s, ws = await _playing(make_session)
    await s._on_vad(VadEvent("start"))
    assert ws.types()[-1] == "output_audio.paused"
    sent = len(ws.audio)
    await asyncio.sleep(0.1)
    assert len(ws.audio) == sent  # nothing is sent while paused
    await s._on_stt(SttUpdate(partial="wait, what"))
    await wait_for(ws, "response.done")
    done = ws.of("response.done")[0]
    assert done["status"] == "cancelled" and done["reason"] == "barge_in"


async def test_cough_resumes_where_it_paused(make_session):
    s, ws = await _playing(make_session, false_interruption_ms=100)
    await s._on_vad(VadEvent("start"))
    sent = len(ws.audio)
    await s._on_vad(VadEvent("stop", 250))
    await wait_for(ws, "output_audio.resumed")
    assert ws.of("output_audio.resumed")[0]["reason"] == "false_interruption"
    await asyncio.sleep(0.1)
    assert len(ws.audio) > sent  # playback continued
    await wait_for(ws, "response.done")
    assert ws.of("response.done")[0]["status"] == "completed"


async def test_long_speech_confirms_without_words(make_session):
    s, ws = await _playing(make_session, interrupt_confirm_ms=300)
    await s._on_vad(VadEvent("start"))
    await s._on_vad_speaking(200)
    assert not ws.of("response.done")
    await s._on_vad_speaking(320)
    await wait_for(ws, "response.done")
    assert ws.of("response.done")[0]["status"] == "cancelled"


async def test_backchannel_during_reply_is_not_a_turn(make_session):
    s, ws = await _playing(make_session)
    await s._on_vad(VadEvent("start"))
    await s._on_stt(SttUpdate(partial="haan"))
    await s._on_vad(VadEvent("stop", 300))
    await s._on_stt(SttUpdate(finals=[Token("Haan.", 3000, 3300, True)], endpoint=True))
    ignored = [e for e in ws.of("transcript.done") if e.get("ignored")]
    assert ignored and ignored[0]["ignored"] == "backchannel"
    assert ws.of("output_audio.resumed")[0]["reason"] == "backchannel"
    await wait_for(ws, "response.done")
    assert ws.of("response.done")[0]["status"] == "completed"
    assert [m["content"] for m in s.history if m["role"] == "user"] == ["Explain photosynthesis"]


async def test_own_echo_does_not_interrupt(make_session):
    s, ws = await _playing(make_session)
    echo = " ".join(s._resp.queued_text().split()[:5])
    await s._on_stt(SttUpdate(partial=echo))
    assert not ws.of("response.done")
    await s._on_stt(SttUpdate(partial=echo + " no stop, explain respiration instead"))
    await wait_for(ws, "response.done")
    assert ws.of("response.done")[0]["status"] == "cancelled"


async def test_reply_stays_interruptible_until_played(make_session):
    # TTS renders 1.2 s of audio almost instantly; playout keeps the reply live while it plays.
    s, ws, *_ = make_session(worker=FakeWorker(chunk_delay=0, chunks=60))
    await s._on_stt(utterance("Explain photosynthesis"))
    await wait_for(ws, "response.audio.started")
    await asyncio.sleep(0.3)
    assert s._resp is not None and not ws.of("response.done")
    await s._on_stt(SttUpdate(partial="stop please"))
    await wait_for(ws, "response.done")
    done = ws.of("response.done")[0]
    assert done["status"] == "cancelled"
    # roughly 0.3 s was heard, so far fewer than all the sentences count as said
    assert len(done["text"]) < len(s.deps.llm.reply)


async def _speak_then_pause(s, words, start=100, end=900):
    """User says `words`; the VAD hears them stop; STT has them as live partial text."""
    await s._on_stt(SttUpdate(partial=words))
    await s._on_vad(VadEvent("start"))
    await s._on_vad(VadEvent("stop", end - start))


def _client_saw_response(ws):
    return [e for e in ws.events if e["type"].startswith("response.")]


async def test_draft_is_held_then_released_on_matching_final(make_session):
    s, ws, *_ = make_session()
    await _speak_then_pause(s, "What is photosynthesis")
    assert s._resp is not None and s._resp.held
    await asyncio.sleep(0.15)  # the draft generates and renders meanwhile
    assert _client_saw_response(ws) == [] and len(ws.audio) == 0
    await s._on_stt(utterance("What is photosynthesis?"))
    await wait_for(ws, "response.done")
    started = ws.of("response.started")[0]
    assert started["turn_id"] == "turn_1"
    done = ws.of("response.done")[0]
    assert done["status"] == "completed" and "first_audio_after_turn_end" in done["timings"]
    assert len(s.deps.llm.calls) == 1  # no second LLM call: the draft was the reply
    assert [m["role"] for m in s.history] == ["user", "assistant"]


async def test_draft_discarded_unseen_when_user_keeps_talking(make_session):
    s, ws, *_ = make_session()
    await _speak_then_pause(s, "What is")
    draft = s._resp
    await s._on_vad(VadEvent("start"))  # they weren't done
    assert draft.task.done() and s._resp is None
    await s._on_vad(VadEvent("stop", 900))
    await s._on_stt(utterance("What is photosynthesis exactly?"))
    await wait_for(ws, "response.done")
    assert len(ws.of("response.started")) == 1 and ws.of("response.done")[0]["status"] == "completed"
    assert s.deps.llm.calls[-1][-1]["content"].startswith("What is photosynthesis exactly?")


async def test_draft_replaced_when_final_words_differ(make_session):
    s, ws, *_ = make_session()
    await _speak_then_pause(s, "What is photo since this")
    await s._on_stt(utterance("Why do leaves change colour in autumn?"))
    await wait_for(ws, "response.done")
    assert len(ws.of("response.started")) == 1
    assert s.deps.llm.calls[-1][-1]["content"].startswith("Why do leaves")


async def test_late_words_after_pause_still_get_a_draft_and_redraft(make_session):
    s, ws, *_ = make_session()
    await s._on_vad(VadEvent("start"))
    await s._on_vad(VadEvent("stop", 800))
    assert s._resp is None  # the pause came before any words
    await s._on_stt(SttUpdate(partial="What is"))
    first = s._resp
    assert first is not None and first.held and first.draft_text == "What is"
    await s._on_stt(SttUpdate(partial="What is photosynthesis"))
    assert first.task.done() and s._resp.draft_text == "What is photosynthesis"
    await s._on_stt(utterance("What is photosynthesis?"))
    await wait_for(ws, "response.done")
    assert ws.of("response.done")[0]["status"] == "completed" and len(ws.of("response.started")) == 1


async def test_voice_persona_and_greeting(make_session):
    s, ws, db, cust = make_session()
    db.set_persona(s.voice["id"], "You are Ritesh Sir, warm and energetic.", "नमस्ते बच्चों!")
    s.cfg.voice_id = None  # let session.update set it again, as a client would
    await s._on_event({"type": "session.update", "session": {"voice_id": s.voice["id"], "greet": True}})
    await wait_for(ws, "response.done")
    assert ws.of("response.done")[0]["text"] == "नमस्ते बच्चों!"
    await s._on_event({"type": "session.update", "session": {"greet": True}})  # greets only once
    await s._on_stt(utterance("Sir, what is DNA?"))
    await wait_for(ws, "response.done", 2)
    system, *rest = s.deps.llm.calls[-1]
    assert system["content"].endswith("You are Ritesh Sir, warm and energetic.")
    assert rest[0] == {"role": "assistant", "content": "नमस्ते बच्चों!"}
    assert len(ws.of("response.started")) == 2


class _CountingStt:
    provider, model, needs_endpointing, offset_ms, audio_bytes = "fake", "m", True, 0, 0

    def __init__(self):
        self.finalized = 0

    async def finalize(self):
        self.finalized += 1


class _FixedTurn:
    def __init__(self, p):
        self.p = p

    def p_complete(self, audio):
        return self.p


async def test_confident_turn_end_finalizes_at_the_pause(make_session):
    s, ws, *_ = make_session()
    s.stt, s._turn_detector = _CountingStt(), _FixedTurn(0.9)
    await s._on_stt(SttUpdate(partial="What is DNA"))
    await s._on_vad(VadEvent("start"))
    await s._on_vad(VadEvent("pause", 900))
    for _ in range(50):
        if s.stt.finalized:
            break
        await asyncio.sleep(0.01)
    assert s.stt.finalized == 1 and s._turn_finalized
    assert s._resp is not None and s._resp.held  # the draft starts at the pause, not at the stop


async def test_unsure_turn_end_waits_for_silence(make_session):
    s, ws, *_ = make_session()
    s.stt, s._turn_detector = _CountingStt(), _FixedTurn(0.05)
    await s._on_vad(VadEvent("start"))
    await s._on_vad(VadEvent("pause", 900))
    await asyncio.sleep(0.05)
    assert s.stt.finalized == 0 and s._turn_p == 0.05


async def test_speaking_again_after_pause_drops_the_draft(make_session):
    s, ws, *_ = make_session()
    s.stt, s._turn_detector = _CountingStt(), _FixedTurn(0.9)
    await s._on_stt(SttUpdate(partial="What is DNA"))
    await s._on_vad(VadEvent("start"))
    await s._on_vad(VadEvent("pause", 900))
    await asyncio.sleep(0.05)
    draft = s._resp
    assert draft is not None and draft.held
    await s._on_vad(VadEvent("resume"))
    assert draft.task.done() and s._resp is None


async def test_reply_opening_with_a_cached_opener_skips_tts_for_it(make_session):
    from voicegw import session as sess

    llm = FakeLLM(reply="अच्छा, DNA एक molecule है जो genetic information रखता है। इसे समझना आसान है।")
    s, ws, *_ = make_session(llm=llm)
    key = (s.voice["id"], s.cfg.output_sample_rate)
    sess._OPENER_CACHE[key] = {o: b"\x02\x00" * 4800 for o in sess.OPENERS}
    try:
        await s._on_stt(utterance("DNA kya hai?"))
        await wait_for(ws, "response.done")
        done = ws.of("response.done")[0]
        assert done["status"] == "completed" and "opener" in done["timings"]
        assert not s.deps.worker.spoken[0].startswith("अच्छा")  # the opener came from the cache
        assert ws.audio[:4] == b"\x02\x00\x02\x00"  # and it played first
        assert sess.OPENER_RULE in llm.calls[0][0]["content"]
    finally:
        sess._OPENER_CACHE.pop(key, None)
