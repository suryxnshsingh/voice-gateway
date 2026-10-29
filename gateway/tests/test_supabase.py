"""Supabase accounts: keys, voices, credit and the usage log, against a fake PostgREST."""

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from voicegw.accounts import SupabaseAccounts, UsageSync
from voicegw.config import settings
from voicegw.db import DB

from test_public_api import FakeTranscriber, FakeWorker

USER = "6f1c2b0e-0d5a-4d4e-9a57-2f4b8f0f3a11"
KEY = "vg_" + "ab" * 24
SESSION = "eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiI2ZjFjIn0.c2lnbmF0dXJl"  # a dashboard access token (shape only)


class FakePostgrest:
    """Just the calls the gateway makes, answered the way PostgREST answers them."""

    def __init__(self):
        self.status = "active"
        self.role = "user"
        self.revoked_at = None
        self.balance = 50000
        self.down = False
        self.refuse = set()          # request_ids usage_events rejects (as a foreign key error would)
        self.usage: dict[str, dict] = {}
        self.calls: list[str] = []
        self.voices = [
            {"id": "voice_mine123", "user_id": USER, "name": "Ritesh Sir", "speaker_name": "Ritesh",
             "status": "active", "reference_seconds": "8.0", "created_at": "2026-10-27T10:00:00+00:00",
             "activated_at": "2026-10-27T11:00:00.5+00:00"},
            {"id": "voice_shared1", "user_id": None, "name": "Kavya", "speaker_name": "Kavya", "status": "active",
             "reference_seconds": "30.5", "created_at": "2026-10-26T10:00:00+00:00", "activated_at": None},
            {"id": "voice_tuned01", "user_id": USER, "name": "Ritesh Sir, fine tuned", "speaker_name": "Ritesh",
             "status": "active", "reference_seconds": "1320.0", "adapter": "ritesh_step250",
             "created_at": "2026-10-28T10:00:00+00:00", "activated_at": "2026-10-28T11:00:00+00:00"},
        ]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] == "sb_secret_test"
        path = request.url.path.removeprefix("/rest/v1")
        self.calls.append(path)
        if self.down:
            raise httpx.ConnectError("down")
        if path == "/rpc/verify_api_key":
            ok = json.loads(request.content)["p_key"] == KEY and self.revoked_at is None
            return httpx.Response(200, json=[{"user_id": USER, "api_key_id": 7, "plan": "payg", "status": self.status}] if ok else [])
        if path == "/auth/v1/user":
            ok = request.headers["authorization"] == f"Bearer {SESSION}"
            return httpx.Response(200, json={"id": USER, "aud": "authenticated"}) if ok else httpx.Response(
                403, json={"code": "bad_jwt", "message": "invalid JWT"})
        if path == "/profiles":
            assert request.url.params["id"] == f"eq.{USER}"
            return httpx.Response(200, json=[{"status": self.status, "role": self.role}])
        if path == "/api_keys":
            assert request.url.params["user_id"] == f"eq.{USER}"
            if request.url.params["id"] != "eq.7":
                return httpx.Response(200, json=[])  # not one of this account's keys
            return httpx.Response(200, json=[{"revoked_at": self.revoked_at, "profiles": {"status": self.status}}])
        if path == "/voices":
            assert request.url.params["or"] == f"(user_id.eq.{USER},user_id.is.null)"
            return httpx.Response(200, json=self.voices)
        if path == "/rpc/credit_balance":
            return httpx.Response(200, json=[{"credits_paise": 0, "spent_paise": 0, "balance_paise": self.balance}])
        if path == "/usage_events":
            assert request.url.params["on_conflict"] == "request_id"
            assert "ignore-duplicates" in request.headers["prefer"]
            rows = json.loads(request.content)
            if any(r["request_id"] in self.refuse for r in rows):
                return httpx.Response(409, json={"code": "23503", "message": "violates foreign key constraint"})
            for r in rows:
                self.usage.setdefault(r["request_id"], r)
            return httpx.Response(201)
        return httpx.Response(404, json={"message": f"no route {path}"})


@pytest.fixture
def sb(tmp_path, monkeypatch):
    from voicegw.app import app

    with TestClient(app) as client:
        monkeypatch.setattr(settings, "data_dir", tmp_path)
        fake = FakePostgrest()
        db = DB(tmp_path / "t.sqlite3")
        accounts = SupabaseAccounts("https://proj.supabase.co", "sb_secret_test", cache_s=30,
                                    transport=httpx.MockTransport(fake))
        app.state.db, app.state.accounts = db, accounts
        app.state.worker, app.state.transcriber = FakeWorker(), FakeTranscriber()
        for vid in ("voice_mine123", "voice_shared1", "voice_tuned01"):
            (tmp_path / "voices" / vid).mkdir(parents=True)
            (tmp_path / "voices" / vid / "reference.wav").write_bytes(b"RIFF")
        yield client, {"Authorization": f"Bearer {KEY}"}, fake, db, UsageSync(db, accounts.http)


def test_voices_come_from_supabase(sb):
    client, auth, fake, _, _ = sb
    voices = client.get("/v1/voices", headers=auth).json()["voices"]
    assert [v["id"] for v in voices] == ["voice_mine123", "voice_shared1", "voice_tuned01"]
    assert voices[0]["shared"] is False and voices[1]["shared"] is True
    assert voices[0]["reference_seconds"] == 8.0 and voices[0]["activated_at"] == 1793098800.5
    assert client.get("/v1/voices/voice_shared1", headers=auth).status_code == 200
    assert client.get("/v1/voices/voice_other", headers=auth).status_code == 404
    client.get("/v1/voices", headers=auth)
    assert fake.calls.count("/rpc/verify_api_key") == 1  # cached


def test_unknown_key_and_inactive_accounts(sb):
    client, auth, fake, _, _ = sb
    assert client.get("/v1/voices", headers={"Authorization": "Bearer vg_nope"}).status_code == 401
    fake.status = "suspended"
    r = client.get("/v1/models", headers=auth)
    assert r.status_code == 403 and "suspended" in r.json()["detail"]


async def test_usage_rows_reach_supabase(sb):
    client, auth, fake, db, sync = sb
    r = client.post("/v1/tts", headers=auth, json={"voice_id": "voice_mine123", "text": "नमस्ते बेटा।", "format": "pcm"})
    assert r.status_code == 200
    assert client.post("/v1/tts", headers=auth, json={"voice_id": "voice_nope", "text": "hi"}).status_code == 404
    stt = client.post("/v1/stt", headers=auth, files={"file": ("q.wav", b"audio", "audio/wav")})
    assert stt.status_code == 200
    assert await sync.flush() == 3 and await sync.flush() == 0
    rows = {r["endpoint"] + str(r["status_code"]): r for r in fake.usage.values()}
    tts = rows["tts200"]
    assert tts["request_id"] == r.headers["x-request-id"] and tts["user_id"] == USER and tts["api_key_id"] == 7
    assert tts["characters"] == len("नमस्ते बेटा।") and tts["audio_seconds"] == 0.06 and tts["latency_ms"] is not None
    assert rows["tts404"]["voice_id"] == "voice_nope"
    assert rows["stt200"]["audio_seconds"] == 3.2 and rows["stt200"]["characters"] == 0
    assert db.request_counts() == {"sent": 3}


async def test_a_refused_row_does_not_block_the_rest(sb):
    client, auth, fake, db, sync = sb
    ids = [client.post("/v1/tts", headers=auth, json={"voice_id": "voice_mine123", "text": f"वाक्य {i}।"})
           .headers["x-request-id"] for i in range(3)]
    fake.refuse = {ids[1]}
    assert await sync.flush() == 3
    assert set(fake.usage) == {ids[0], ids[2]} and db.request_counts() == {"sent": 2, "failed": 1}


async def test_supabase_outage(sb):
    client, auth, fake, db, sync = sb
    assert client.get("/v1/voices", headers=auth).status_code == 200
    fake.down = True
    accounts = client.app.state.accounts
    accounts._cache = {k: (time.monotonic() - 1, v[1]) for k, v in accounts._cache.items()}  # just expired: still used
    assert client.post("/v1/tts", headers=auth, json={"voice_id": "voice_mine123", "text": "हाँ।"}).status_code == 200
    with pytest.raises(Exception):
        await sync.flush()  # stays queued
    assert db.request_counts() == {"pending": 1}
    other = client.get("/v1/models", headers={"Authorization": "Bearer vg_" + "cd" * 24})
    assert other.status_code == 503  # nothing cached for this key
    fake.down = False
    assert await sync.flush() == 1


def test_credit_is_enforced_when_enabled(sb, monkeypatch):
    client, auth, fake, db, _ = sb
    fake.balance = 0
    assert client.post("/v1/tts", headers=auth, json={"voice_id": "voice_mine123", "text": "हाँ।"}).status_code == 200
    monkeypatch.setattr(settings, "require_credit", True)
    r = client.post("/v1/tts", headers=auth, json={"voice_id": "voice_mine123", "text": "हाँ।"})
    assert r.status_code == 402 and r.headers["x-request-id"].startswith("req_")
    assert client.post("/v1/stt", headers=auth, files={"file": ("q.wav", b"a", "audio/wav")}).status_code == 402


def test_voice_live_in_supabase_but_missing_here(sb, tmp_path):
    client, auth, _, _, _ = sb
    (tmp_path / "voices" / "voice_shared1" / "reference.wav").unlink()
    assert client.post("/v1/tts", headers=auth, json={"voice_id": "voice_shared1", "text": "हाँ।"}).status_code == 503


def test_browser_token_for_the_stream(sb):
    client, auth, fake, db, _ = sb
    token = client.post("/v1/tokens", headers=auth).json()["token"]
    with client.websocket_connect(f"/v1/tts/stream?voice_id=voice_mine123&token={token}") as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text(json.dumps({"type": "text", "text": "पहला वाक्य पूरा हुआ। "}))
        ws.send_text(json.dumps({"type": "close"}))
        while not (m := ws.receive()).get("text") or json.loads(m["text"])["type"] != "done":
            pass
    rows = db.pending_requests(10)
    assert [(r["endpoint"], r["status_code"], r["api_key_id"]) for r in rows] == [("tts_stream", 200, 7)]
    assert rows[0]["characters"] == len("पहला वाक्य पूरा हुआ।")
    fake.revoked_at = "2026-10-28T10:00:00+00:00"
    client.app.state.accounts._cache.clear()
    with pytest.raises(Exception):
        with client.websocket_connect(f"/v1/tts/stream?voice_id=voice_mine123&token={token}") as ws:
            ws.receive_json()


def test_stream_refusals_carry_their_close_codes(sb):
    from starlette.websockets import WebSocketDisconnect

    client, auth, fake, _, _ = sb
    for url, code in [("/v1/tts/stream?voice_id=voice_mine123&token=vgs_bad.sig", 4401),
                      ("/v1/tts/stream?voice_id=voice_mine123&sample_rate=32000", 4400)]:
        with pytest.raises(WebSocketDisconnect) as e:
            with client.websocket_connect(url, headers=auth if "sample_rate" in url else {}) as ws:
                ws.receive_json()
        assert e.value.code == code
    fake.status = "pending"
    client.app.state.accounts._cache.clear()
    with pytest.raises(WebSocketDisconnect) as e:
        with client.websocket_connect("/v1/tts/stream?voice_id=voice_mine123", headers=auth) as ws:
            ws.receive_json()
    assert e.value.code == 4403


def test_stream_engine_failure_reaches_the_client(sb):
    from voicegw.providers.worker import WorkerError

    client, auth, _, db, _ = sb

    async def broken(voice_id, text, sr, result, timesteps=None):
        raise WorkerError("tts_failed")
        yield b""

    client.app.state.worker.tts = broken
    with client.websocket_connect("/v1/tts/stream?voice_id=voice_mine123", headers=auth) as ws:
        assert ws.receive_json()["type"] == "ready"
        ws.send_text(json.dumps({"type": "text", "text": "यह वाक्य टूटेगा। "}))
        ws.send_text(json.dumps({"type": "flush"}))
        assert ws.receive_json() == {"type": "error", "message": "Speech generation failed."}
    assert [(r["endpoint"], r["status_code"]) for r in db.pending_requests(5)] == [("tts_stream", 500)]


async def test_dashboard_session_works_like_a_key(sb):
    """The dashboard's playground calls the API with the signed in session, billed to a key it names."""
    client, _, fake, db, sync = sb
    session = {"Authorization": f"Bearer {SESSION}", "X-Api-Key-Id": "7"}
    assert [v["id"] for v in client.get("/v1/voices", headers=session).json()["voices"]] == [
        "voice_mine123", "voice_shared1", "voice_tuned01"]
    r = client.post("/v1/tts", headers=session, json={"voice_id": "voice_mine123", "text": "नमस्ते बेटा।", "format": "pcm",
                                                      "stream": True})
    assert r.status_code == 200 and len(r.content) > 0
    assert await sync.flush() == 1
    row = next(iter(fake.usage.values()))
    assert row["user_id"] == USER and row["api_key_id"] == 7 and row["endpoint"] == "tts"
    bad = client.get("/v1/voices", headers={"Authorization": "Bearer eyJx.eyJy.expired"})
    assert bad.status_code == 401
    fake.status = "pending"
    client.app.state.accounts._cache.clear()
    r = client.get("/v1/voices", headers=session)
    assert r.status_code == 403 and "approval" in r.json()["detail"]


def test_a_customer_session_needs_one_of_their_keys(sb):
    client, _, fake, _, _ = sb
    bare = {"Authorization": f"Bearer {SESSION}"}
    r = client.post("/v1/tts", headers=bare, json={"voice_id": "voice_mine123", "text": "हाँ।"})
    assert r.status_code == 400 and "API keys" in r.json()["detail"]
    other = client.post("/v1/tts", headers={**bare, "X-Api-Key-Id": "8"}, json={"voice_id": "voice_mine123", "text": "हाँ।"})
    assert other.status_code == 403
    assert client.get("/v1/voices", headers={**bare, "X-Api-Key-Id": "seven"}).status_code == 400
    fake.revoked_at = "2026-10-29T10:00:00+00:00"
    client.app.state.accounts._cache.clear()
    revoked = client.get("/v1/voices", headers={**bare, "X-Api-Key-Id": "7"})
    assert revoked.status_code == 403 and "revoked" in revoked.json()["detail"]


async def test_an_admin_session_without_a_key_is_free(sb, monkeypatch):
    """Admins try things in the playground without a key: no credit check, and no key on the usage row,
    which Supabase prices at 0."""
    client, _, fake, _, sync = sb
    fake.role, fake.balance = "admin", 0
    monkeypatch.setattr(settings, "require_credit", True)
    admin = {"Authorization": f"Bearer {SESSION}"}
    r = client.post("/v1/tts", headers=admin, json={"voice_id": "voice_mine123", "text": "हाँ।", "format": "pcm"})
    assert r.status_code == 200
    assert client.post("/v1/stt", headers=admin, files={"file": ("q.wav", b"audio", "audio/wav")}).status_code == 200
    assert await sync.flush() == 2
    assert {row["api_key_id"] for row in fake.usage.values()} == {None}
    # With a key, an admin is billed like anyone else.
    keyed = client.post("/v1/tts", headers={**admin, "X-Api-Key-Id": "7"}, json={"voice_id": "voice_mine123", "text": "हाँ।"})
    assert keyed.status_code == 402


def test_a_fine_tuned_voice_renders_with_its_adapter_and_hides_it(sb):
    """The worker gets the adapter; the API only says the voice is fine tuned."""
    client, auth, fake, _, _ = sb
    voices = {v["id"]: v for v in client.get("/v1/voices", headers=auth).json()["voices"]}
    assert voices["voice_tuned01"]["fine_tuned"] is True and voices["voice_mine123"]["fine_tuned"] is False
    assert all("adapter" not in v for v in voices.values())
    assert "adapter" not in client.get("/v1/voices/voice_tuned01", headers=auth).json()
    worker = client.app.state.worker
    worker.adapters = []
    client.post("/v1/tts", headers=auth, json={"voice_id": "voice_tuned01", "text": "नमस्ते बेटा।", "format": "pcm"})
    client.post("/v1/tts", headers=auth, json={"voice_id": "voice_mine123", "text": "नमस्ते बेटा।", "format": "pcm"})
    assert worker.adapters == ["ritesh_step250", None]
