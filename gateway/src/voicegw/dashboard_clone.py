"""Cloned voices made automatically from the Dhvni dashboard.

A customer starts a cloned voice request in the dashboard and uploads, to the voice-requests bucket in
Supabase Storage, recordings of the speaker and the speaker reading the consent phrase with three code
words the database picked. This endpoint checks the consent the way consent.py does for self serve
voices: every code word must be in the transcript, so the recording was made for this request, and the
consent voice must be the same speaker as the reference. If both pass, the reference is stored here,
loaded on the worker, and Supabase adds the voice and marks the request live (complete_clone_request,
which rechecks the plan's allowance). A cloned voice runs zero shot on the base model.

Fine-tuned voices never come through here: the team trains those by hand and links their adapter.
"""

import json
import logging
import shutil
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from . import consent
from .accounts import SupabaseError
from .audio import AudioError, ffmpeg_convert, pcm_seconds, wav_bytes
from .auth import bearer, looks_like_session
from .config import settings
from .db import new_id
from .providers.transcribe import TranscribeError

log = logging.getLogger(__name__)
router = APIRouter()

# Zero shot needs seconds of speech, not minutes; a longer prompt only slows every sentence it speaks.
REFERENCE_MAX_S = 30.0
REFERENCE_SR = 48000
MAX_FILE_BYTES = 50 * 1024 * 1024  # the bucket's own limit


def _supabase_message(e: SupabaseError) -> str:
    """The sentence a database check raised (for example the plan's allowance), for the customer."""
    try:
        return json.loads(str(e).split(": ", 1)[1]).get("message") or "That did not work. Please retry."
    except (IndexError, ValueError, AttributeError):
        return "That did not work. Please retry."


@router.post("/v1/dashboard/voice-requests/{request_id}/clone", include_in_schema=False)
async def clone_voice(request_id: int, request: Request):
    app = request.app
    accounts = app.state.accounts
    if not getattr(accounts, "remote", False):
        raise HTTPException(404, "Not Found")
    token = bearer(request.headers.get("authorization"))
    if not token or not looks_like_session(token):
        raise HTTPException(401, "Sign in to the dashboard first.")
    user = await accounts.session_user(token)
    if user is None:
        raise HTTPException(401, "Your session has ended. Sign in again.")
    user_id, status, _role = user
    if status != "active":
        raise HTTPException(403, "Your account needs to be approved before it can have voices.")

    r = await accounts.clone_request(request_id, user_id)
    if r is None:
        raise HTTPException(404, "We could not find that voice request.")
    if r["kind"] != "clone":
        raise HTTPException(400, "Fine tuned voices are trained by our team. Send the request to them instead.")
    if r["consent"] != "phrase":
        raise HTTPException(400, "A cloned voice needs the speaker to read the consent phrase. A signed release goes to our team.")
    if r["status"] not in ("draft", "changes_requested"):
        raise HTTPException(409, "This request is not open to check.")
    if r["auto_attempts"] >= settings.consent_max_attempts:
        raise HTTPException(429, f"This request has been checked {settings.consent_max_attempts} times. "
                                 "Start a new request, or send this one to our team.")
    allowance = await accounts.allowance(user_id)
    limit = allowance.get("clone_limit")
    if limit is not None and allowance.get("clones_used", 0) >= limit:
        raise HTTPException(409, f"Your plan includes {limit} cloned voices, and all are in use. Change your plan to add more.")

    references = await accounts.request_files(user_id, request_id, "reference")
    consents = await accounts.request_files(user_id, request_id, "consent")
    if not references:
        raise HTTPException(400, "Add a recording of the speaker first.")
    if not consents:
        raise HTTPException(400, "Add the consent recording first.")

    # The reference: the speaker's recordings in upload order, joined and cut to REFERENCE_MAX_S.
    pcm = b""
    for f in references:
        if f["size"] > MAX_FILE_BYTES:
            continue
        try:
            pcm += await ffmpeg_convert(await accounts.download(f["path"]), REFERENCE_SR, "s16le")
        except AudioError:
            raise HTTPException(400, f"We could not read {f['name']}. Upload mp3, m4a, wav, webm, ogg or flac.") from None
        if pcm_seconds(pcm, REFERENCE_SR) >= REFERENCE_MAX_S:
            break
    pcm = pcm[: int(REFERENCE_MAX_S * REFERENCE_SR) * 2]
    seconds = pcm_seconds(pcm, REFERENCE_SR)
    if seconds < settings.voice_min_seconds:
        raise HTTPException(400, f"The recordings come to {seconds:.0f} s. Add at least 10 seconds of the speaker talking.")
    reference = wav_bytes(pcm, REFERENCE_SR)

    # The consent: the newest recording of the phrase.
    latest = consents[-1]
    try:
        consent_pcm = await ffmpeg_convert(await accounts.download(latest["path"]), 16000, "s16le")
    except AudioError:
        raise HTTPException(400, f"We could not read {latest['name']}. Record the consent phrase again.") from None
    if pcm_seconds(consent_pcm, 16000) < 2:
        raise HTTPException(400, "The consent recording is too short. Record the whole phrase.")
    consent_wav = wav_bytes(consent_pcm, 16000)

    # The phrase is English, and English transcription keeps the code words in English letters.
    try:
        heard = await app.state.transcriber.transcribe(consent_wav, "consent.wav", "en", None)
    except TranscribeError:
        raise HTTPException(503, "We could not check the consent just now. Please retry in a minute.") from None
    code_ok, missing = consent.code_matches(" ".join(r["consent_words"]), heard.text)
    score = await app.state.worker.speaker_similarity(reference, consent_wav)
    speaker_ok = score >= settings.consent_speaker_threshold
    check = {
        "passed": code_ok and speaker_ok,
        "code_words": {"passed": code_ok, "missing": missing},
        "same_speaker": {"passed": speaker_ok, "score": round(score, 4), "threshold": settings.consent_speaker_threshold},
        "transcript": heard.text,
        "consent_file": latest["name"],
        "reference_seconds": round(seconds, 1),
        "checked_at": int(time.time()),
    }
    public_checks = {"code_words": check["code_words"], "same_speaker": {"passed": speaker_ok}}

    if not check["passed"]:
        attempts = await accounts.record_clone_check(request_id, check)
        log.info("clone check %s failed: missing %s, speaker %.2f", request_id, missing, score)
        return JSONResponse(status_code=422, content={
            "passed": False, "checks": public_checks,
            "attempts_left": max(0, settings.consent_max_attempts - attempts),
        })

    voice_id = new_id("voice")
    vdir = settings.voices_dir / voice_id
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / "reference.wav").write_bytes(reference)
    try:
        await app.state.worker.put_voice(voice_id, reference)
        await accounts.complete_clone_request(request_id, voice_id, seconds, check)
    except Exception as e:
        shutil.rmtree(vdir, ignore_errors=True)
        try:
            await app.state.worker.delete_voice(voice_id)
        except Exception:
            log.warning("could not remove %s from the worker", voice_id)
        if isinstance(e, SupabaseError):
            raise HTTPException(409, _supabase_message(e)) from None
        log.exception("clone %s failed after its check passed", request_id)
        raise HTTPException(503, "The voice could not be set up just now. Please retry.") from None
    accounts.forget_voices(user_id)
    log.info("clone %s: %s live, %.1f s reference, speaker %.2f", request_id, voice_id, seconds, score)
    return {"passed": True, "voice_id": voice_id, "reference_seconds": round(seconds, 1), "checks": public_checks}
