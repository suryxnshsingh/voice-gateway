import io
import logging
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, RedirectResponse

from . import consent
from .audio import AudioError, ffmpeg_convert, pcm_seconds
from .auth import CurrentCustomer, bearer, mint_session_token, read_session_token
from .config import settings
from .db import DB, Customer
from .providers import stt
from .providers.soniox import SttError
from .limits import Limiter
from .providers.llm import LLM
from .providers.transcribe import Transcriber
from .providers.worker import WorkerClient
from .public_api import router as public_router
from .session import Deps, Session

log = logging.getLogger("voicegw")

MAX_UPLOAD = 25 * 1024 * 1024
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.voices_dir.mkdir(parents=True, exist_ok=True)
    app.state.db = DB(settings.db_path)
    app.state.worker = WorkerClient(settings.worker_url, settings.worker_token)
    app.state.llm = LLM(settings.llm_api_key, settings.llm_base_url, settings.llm_service_tier)
    app.state.transcriber = Transcriber(settings.openai_key, settings.stt_relay_model)
    app.state.limiter = Limiter(settings.requests_per_minute,
                                {"tts": settings.tts_concurrency, "stt": settings.stt_concurrency})
    yield
    await app.state.worker.close()


app = FastAPI(
    title=f"{settings.brand} API",
    version="1.0",
    description=(f"Text-to-speech and speech-to-text for Hindi, English and Hinglish.\n\n"
                 f"Authenticate with `Authorization: Bearer <api key>`. Models: `{settings.brand_tts_model}` "
                 f"(text-to-speech), `{settings.brand_stt_model}` (speech-to-text). Streaming text-to-speech: "
                 f"WebSocket `/v1/tts/stream` (see the API guide)."),
    lifespan=lifespan,
    redoc_url=None,
)
app.include_router(public_router)


def _deps(app: FastAPI) -> Deps:
    return Deps(app.state.db, app.state.worker, app.state.llm, settings)


def public_voice(v: dict) -> dict:
    return {
        "id": v["id"],
        "name": v["name"],
        "speaker_name": v["speaker_name"],
        "status": v["status"],
        "shared": v["customer_id"] is None,
        "reference_seconds": v["reference_seconds"],
        "created_at": v["created_at"],
        "activated_at": v["activated_at"],
    }


async def _read_upload(f: UploadFile) -> bytes:
    data = await f.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "file too large (25 MB max)")
    return data


def _wav_seconds(wav: bytes) -> float:
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / w.getframerate()


@app.get("/", include_in_schema=False)
async def root():
    if not settings.playground_enabled:
        return RedirectResponse("/docs")
    return RedirectResponse("/playground")


@app.get("/playground", include_in_schema=False)
async def playground():
    if not settings.playground_enabled:
        raise HTTPException(404, "Not Found")
    return FileResponse(STATIC / "playground.html")


@app.get("/health", tags=["status"], summary="Service status")
async def health(request: Request):
    try:
        await request.app.state.worker.health()
        ok = True
    except Exception:
        ok = False
    return {"ok": ok}


@app.get("/internal/health", include_in_schema=False)
async def internal_health(request: Request):
    """Engine details for operators; only answered on the gateway machine itself."""
    if not request.client or request.client.host not in ("127.0.0.1", "::1"):
        raise HTTPException(404, "Not Found")
    try:
        worker = await request.app.state.worker.health()
    except Exception as e:
        worker = {"error": str(e)}
    return {"ok": "error" not in worker, "worker": worker, "llm": request.app.state.llm.available,
            "stt_providers": settings.stt_providers, "stt_relay": settings.stt_relay_model}


# realtime


@app.post("/v1/realtime/sessions", include_in_schema=False)
async def create_session_token(customer: Customer = CurrentCustomer):
    """For browsers, which can't put an API key on a WebSocket. The token is good for one connect within 2 minutes."""
    token, exp = mint_session_token(customer.id)
    return {"token": token, "expires_at": exp}


@app.websocket("/v1/realtime")
async def realtime(ws: WebSocket, token: str | None = Query(None)):
    db: DB = ws.app.state.db
    customer = None
    key = bearer(ws.headers.get("authorization"))
    if key:
        customer = db.customer_for_key(key)
    elif token:
        cid = read_session_token(token)
        customer = db.customer(cid) if cid else None
    if customer is None:
        await ws.close(code=4401, reason="unauthorized")
        return
    await ws.accept()
    session = Session(ws, customer, _deps(ws.app))
    log.info("session %s opened for %s", session.id, customer.id)
    await session.run()
    log.info("session %s closed", session.id)


# voices


@app.get("/v1/voices", tags=["voices"], summary="List your voices")
async def list_voices(request: Request, customer: Customer = CurrentCustomer):
    return {"voices": [public_voice(v) for v in request.app.state.db.voices_for(customer.id) if v["status"] == "active"]}


@app.get("/v1/voices/{voice_id}", tags=["voices"], summary="Get a voice")
async def get_voice(voice_id: str, request: Request, customer: Customer = CurrentCustomer):
    v = request.app.state.db.voice_for(customer.id, voice_id)
    if v is None:
        raise HTTPException(404, "voice not found")
    return public_voice(v)


def _self_serve_only() -> None:
    if not settings.self_serve_voices:
        raise HTTPException(404, "Not Found")


@app.post("/v1/voices", status_code=201, include_in_schema=settings.self_serve_voices)
async def create_voice(
    request: Request,
    name: str = Form(..., max_length=100),
    speaker_name: str = Form(..., max_length=100),
    language: str = Form("en", pattern="^(en|hi)$"),
    reference: UploadFile = File(...),
    customer: Customer = CurrentCustomer,
):
    """Step 1 of cloning: upload a reference recording. The voice stays unusable until the speaker records consent."""
    _self_serve_only()
    try:
        wav = await ffmpeg_convert(await _read_upload(reference), 48000)
    except AudioError as e:
        raise HTTPException(400, f"could not read reference audio: {e}") from e
    seconds = _wav_seconds(wav)
    if not settings.voice_min_seconds <= seconds <= settings.voice_max_seconds:
        raise HTTPException(
            400, f"reference must be {settings.voice_min_seconds:.0f}-{settings.voice_max_seconds:.0f} s (got {seconds:.1f} s)"
        )
    phrase, code = consent.new_challenge(speaker_name, customer.name, language)
    expires = time.time() + settings.consent_ttl_hours * 3600
    v = request.app.state.db.create_voice(
        customer_id=customer.id, name=name, speaker_name=speaker_name, reference_seconds=round(seconds, 2),
        consent_phrase=phrase, consent_code=code, consent_expires_at=expires,
    )
    vdir = settings.voices_dir / v["id"]
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "reference.wav").write_bytes(wav)
    return {
        "voice": public_voice(v),
        "consent": {
            "phrase": phrase,
            "expires_at": expires,
            "how": f"Record {speaker_name} reading the phrase aloud, in their own voice, then POST the recording "
                   f"to /v1/voices/{v['id']}/consent as form field 'recording'.",
        },
    }


@app.post("/v1/voices/{voice_id}/consent", include_in_schema=settings.self_serve_voices)
async def submit_consent(voice_id: str, request: Request, recording: UploadFile = File(...),
                         customer: Customer = CurrentCustomer):
    """Step 2: the speaker's recorded consent. Activates the voice if the code words and the speaker both match."""
    _self_serve_only()
    db: DB = request.app.state.db
    v = db.voice(voice_id)
    if v is None or v["customer_id"] != customer.id or v["status"] == "revoked":
        raise HTTPException(404, "voice not found")
    if v["status"] != "pending_consent":
        raise HTTPException(409, "voice already has consent")
    if time.time() > v["consent_expires_at"]:
        raise HTTPException(410, "consent phrase expired; create the voice again")
    if v["consent_attempts"] >= settings.consent_max_attempts:
        raise HTTPException(429, "too many consent attempts; create the voice again")
    db.bump_consent_attempts(voice_id)

    raw = await _read_upload(recording)
    try:
        pcm = await ffmpeg_convert(raw, 16000, "s16le")
        wav16 = await ffmpeg_convert(raw, 16000)
    except AudioError as e:
        raise HTTPException(400, f"could not read recording: {e}") from e
    if pcm_seconds(pcm, 16000) < 2:
        raise HTTPException(400, "recording is too short")

    vdir = settings.voices_dir / voice_id
    attempt = v["consent_attempts"] + 1
    consent_file = vdir / f"consent_{attempt}.wav"
    consent_file.write_bytes(wav16)

    try:
        transcript, stt_provider = await stt.transcribe(pcm, settings, ["en", "hi"])
    except SttError as e:
        raise HTTPException(503, f"speech-to-text unavailable: {e}") from e
    code_ok, missing = consent.code_matches(v["consent_code"], transcript)
    reference = (vdir / "reference.wav").read_bytes()
    score = await request.app.state.worker.speaker_similarity(reference, wav16)
    speaker_ok = score >= settings.consent_speaker_threshold
    passed = code_ok and speaker_ok
    db.add_consent(voice_id, "recorded_statement", passed, {
        "phrase": v["consent_phrase"], "transcript": transcript, "missing_code_words": missing,
        "speaker_score": round(score, 4), "speaker_threshold": settings.consent_speaker_threshold,
        "recording": consent_file.name, "stt_provider": stt_provider,
        "client": request.client.host if request.client else None,
    })
    if passed:
        await request.app.state.worker.put_voice(voice_id, reference)
        db.activate_voice(voice_id)
    return {
        "passed": passed,
        "voice": public_voice(db.voice(voice_id)),
        "checks": {
            "code_words": {"passed": code_ok, "missing": missing},
            "same_speaker": {"passed": speaker_ok},
            "transcript": transcript,
        },
        "attempts_left": settings.consent_max_attempts - attempt,
    }


@app.delete("/v1/voices/{voice_id}", include_in_schema=settings.self_serve_voices)
async def delete_voice(voice_id: str, request: Request, customer: Customer = CurrentCustomer):
    """Revoke a clone. Reference audio is deleted; consent records stay for audit."""
    _self_serve_only()
    db: DB = request.app.state.db
    v = db.voice(voice_id)
    if v is None or v["customer_id"] != customer.id or v["status"] == "revoked":
        raise HTTPException(404, "voice not found")
    db.revoke_voice(voice_id)
    (settings.voices_dir / voice_id / "reference.wav").unlink(missing_ok=True)
    try:
        await request.app.state.worker.delete_voice(voice_id)
    except Exception:
        log.exception("worker delete failed for %s; it will 404 on next use", voice_id)
    return {"deleted": voice_id}


# usage


PUBLIC_USAGE = {"tts_characters": "characters", "tts_seconds": "seconds", "stt_seconds": "seconds"}


@app.get("/v1/usage", tags=["usage"], summary="Your usage")
async def usage(request: Request, since: float | None = None, until: float | None = None,
                customer: Customer = CurrentCustomer):
    """Totals per metered unit (unix-second window, default the last 30 days)."""
    until = until or time.time()
    since = since or until - 30 * 86400
    totals: dict[str, float] = {}
    for row in request.app.state.db.usage_summary(customer.id, since, until):
        if row["kind"] in PUBLIC_USAGE:  # engines and internal agent metering stay internal
            totals[row["kind"]] = totals.get(row["kind"], 0) + row["quantity"]
    return {"since": since, "until": until,
            "usage": [{"metric": k, "quantity": round(v, 2), "unit": PUBLIC_USAGE[k]} for k, v in sorted(totals.items())]}
