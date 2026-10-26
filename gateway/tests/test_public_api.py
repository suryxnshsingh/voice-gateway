"""The v1 public API: TTS and STT endpoints, branding, limits and metering, with fake engines."""

import json

import pytest
from fastapi.testclient import TestClient

from voicegw.config import settings
from voicegw.db import DB
from voicegw.limits import Limiter
from voicegw.providers.transcribe import TranscribeError, Transcript

ENGINE_WORDS = ("openai", "gpt", "voxcpm", "soniox", "whisper")


class FakeWorker:
    def __init__(self):
        self.calls = []

    async def has_voice(self, voice_id):
        return True

    async def close(self):
        pass

    async def health(self):
        return {"tts": {"model": "internal-model-path"}}

    async def tts(self, voice_id, text, sr, result, timesteps=None):
        self.calls.append((text, sr, timesteps))
        for _ in range(3):
            yield b"\x10\x00" * (sr // 50)  # 20 ms
        result.status = "done"


class FakeTranscriber:
    available = True

    def __init__(self, fail=None):
        self.fail, self.calls = fail, []

    async def transcribe(self, data, filename, language, prompt):
        self.calls.append((len(data), filename, language, prompt))
        if self.fail:
            raise self.fail
        return Transcript(text="सर, mitochondria क्या है?", seconds=3.2)


@pytest.fixture
def api(tmp_path):
    from voicegw.app import app

    with TestClient(app) as client:
        db = DB(tmp_path / "t.sqlite3")
        app.state.db, app.state.worker, app.state.transcriber = db, FakeWorker(), FakeTranscriber()
        app.state.limiter = Limiter(1000, {"tts": 4, "stt": 4})
        cust = db.create_customer("Acme")
        key = db.create_api_key(cust.id)
        voice = db.create_voice(customer_id=cust.id, name="Ritesh Sir", speaker_name="Ritesh", reference_seconds=8,
                                consent_phrase=None, consent_code=None, consent_expires_at=None, status="active")
        yield client, {"Authorization": f"Bearer {key}"}, voice, app


def test_models_and_voices_are_branded(api):
    client, auth, voice, _ = api
    models = client.get("/v1/models", headers=auth).json()["models"]
    assert [m["id"] for m in models] == [settings.brand_tts_model, settings.brand_stt_model]
    voices = client.get("/v1/voices", headers=auth).json()["voices"]
    assert [v["id"] for v in voices] == [voice["id"]]


def test_tts_file_and_metering(api):
    client, auth, voice, app = api
    r = client.post("/v1/tts", headers=auth, json={"voice_id": voice["id"], "text": "देखो बेटा, revision ज़रूरी है।",
                                                   "format": "wav", "quality": "hq"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    assert r.headers["x-request-id"].startswith("req_") and float(r.headers["x-audio-seconds"]) == 0.06
    assert app.state.worker.calls[0][2] == 10  # hq = 10 diffusion steps
    usage = client.get("/v1/usage", headers=auth).json()["usage"]
    assert {u["metric"] for u in usage} == {"tts_characters", "tts_seconds"}
    assert all(set(u) == {"metric", "quantity", "unit"} for u in usage)  # no engine/provider fields


def test_tts_mp3_is_encoded(api):
    client, auth, voice, _ = api
    r = client.post("/v1/tts", headers=auth, json={"voice_id": voice["id"], "text": "Namaste bachcho!", "format": "mp3"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/mpeg" and len(r.content) > 100


def test_tts_streaming_pcm(api):
    client, auth, voice, _ = api
    with client.stream("POST", "/v1/tts", headers=auth,
                       json={"voice_id": voice["id"], "text": "पहला वाक्य। दूसरा वाक्य।", "format": "pcm", "stream": True}) as r:
        body = b"".join(r.iter_bytes())
    assert r.status_code == 200 and len(body) == 3 * 2 * (24000 // 50)


def test_tts_errors(api):
    client, auth, voice, _ = api
    assert client.post("/v1/tts", headers=auth, json={"voice_id": "voice_nope", "text": "hi"}).status_code == 404
    assert client.post("/v1/tts", headers=auth, json={"voice_id": voice["id"], "text": "hi", "model": "x"}).status_code == 400
    long = client.post("/v1/tts", headers=auth, json={"voice_id": voice["id"], "text": "a" * (settings.tts_max_chars + 1)})
    assert long.status_code == 413
    assert client.post("/v1/tts", json={"voice_id": voice["id"], "text": "hi"}).status_code == 401


def test_stt_relay_is_branded(api):
    client, auth, _, app = api
    r = client.post("/v1/stt", headers=auth, files={"file": ("q.m4a", b"fake-audio-bytes", "audio/mp4")},
                    data={"language": "hi", "prompt": "NCERT biology"})
    body = r.json()
    assert r.status_code == 200 and body["text"] == "सर, mitochondria क्या है?" and body["model"] == settings.brand_stt_model
    assert body["duration_seconds"] == 3.2
    assert app.state.transcriber.calls == [(16, "q.m4a", "hi", "NCERT biology")]  # the file goes through as-is
    assert not any(w in r.text.lower() for w in ENGINE_WORDS)


def test_stt_errors_are_neutral(api):
    client, auth, _, app = api
    app.state.transcriber = FakeTranscriber(fail=TranscribeError(429, "Transcription is busy right now. Please retry shortly."))
    r = client.post("/v1/stt", headers=auth, files={"file": ("q.wav", b"x", "audio/wav")})
    assert r.status_code == 429 and not any(w in r.text.lower() for w in ENGINE_WORDS)
    assert client.post("/v1/stt", headers=auth, files={"file": ("q.wav", b"", "audio/wav")}).status_code == 400


def test_concurrency_limit(api):
    client, auth, voice, app = api
    app.state.limiter = Limiter(1000, {"tts": 0, "stt": 0})
    r = client.post("/v1/tts", headers=auth, json={"voice_id": voice["id"], "text": "hi there"})
    assert r.status_code == 429 and "retry-after" in r.headers


def test_tts_websocket(api):
    client, auth, voice, _ = api
    with client.websocket_connect(f"/v1/tts/stream?voice_id={voice['id']}", headers=auth) as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text(json.dumps({"type": "text", "text": "Hello बच्चों, आज हम cells पढ़ेंगे। "}))
        ws.send_text(json.dumps({"type": "close"}))
        frames, events = 0, []
        while True:
            msg = ws.receive()
            if msg.get("bytes"):
                frames += 1
            elif msg.get("text"):
                events.append(json.loads(msg["text"])["type"])
                if events[-1] == "done":
                    break
        assert frames == 3 and events == ["chunk.done", "done"]


def test_self_serve_cloning_and_internals_are_hidden(api):
    client, auth, voice, _ = api
    r = client.post("/v1/voices", headers=auth, data={"name": "x", "speaker_name": "y"},
                    files={"reference": ("r.wav", b"x", "audio/wav")})
    assert r.status_code == 404
    assert client.get("/health").json() == {"ok": True}
    assert "/v1/realtime/sessions" not in client.get("/openapi.json").json()["paths"]
