"""Limits for the dashboard's playground.

The playground calls the public API with a signed in dashboard session instead of an API key
(auth.customer_from). It is there to hear a voice on a few short lines, so a session gets far less
than a key: short texts and recordings, a small daily allowance per account, a few requests a
minute, one at a time, and only POST /v1/tts, POST /v1/stt and GET /v1/playground. All the numbers
are in config.py (playground_*). Keys are unaffected.

Allowances are counted from this gateway's own request log (db.requests, source 'playground'),
which every request writes before its slot is released, so the next request already sees it. Days
are India days. An admin without a key (Customer.free) has no daily allowance.
"""

import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request

from .audio import AudioError, ffmpeg_convert, pcm_seconds
from .auth import CurrentCustomer
from .config import settings
from .db import Customer

router = APIRouter()

IST = timezone(timedelta(hours=5, minutes=30))


def day_start(now: float | None = None) -> float:
    """The unix time the current India day began."""
    d = datetime.fromtimestamp(now if now is not None else time.time(), IST)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def limiter(app, customer: Customer):
    """The playground's own limiter (a few a minute, one at a time) for a session; the API's for a key."""
    return app.state.playground_limiter if customer.playground else app.state.limiter


def _used(app, customer: Customer) -> dict[str, float]:
    return app.state.db.playground_used(customer.id, day_start())


def allowance(app, customer: Customer) -> dict:
    """What a session may still do today, as the dashboard shows it. None means no daily limit."""
    used = None if customer.free else _used(app, customer)
    return {
        "tts": {
            "max_characters": settings.playground_tts_max_chars,
            "characters_per_day": None if used is None else settings.playground_tts_chars_per_day,
            "characters_left": None if used is None else max(0, settings.playground_tts_chars_per_day - used["characters"]),
        },
        "stt": {
            "max_seconds": settings.playground_stt_max_seconds,
            "max_bytes": settings.playground_stt_max_bytes,
            "seconds_per_day": None if used is None else settings.playground_stt_seconds_per_day,
            "seconds_left": None if used is None else round(max(0.0, settings.playground_stt_seconds_per_day - used["stt_seconds"]), 1),
        },
        "resets_at": day_start() + 86400,
    }


def check_tts_text(text: str) -> None:
    if len(text) > settings.playground_tts_max_chars:
        raise HTTPException(413, f"The playground speaks up to {settings.playground_tts_max_chars} characters at a time. "
                                 "For longer text, call the API with a key.")


def check_tts_allowance(app, customer: Customer, characters: int) -> None:
    if customer.free:
        return
    left = settings.playground_tts_chars_per_day - _used(app, customer)["characters"]
    if characters > left:
        raise HTTPException(429, f"That is more than today's playground allowance: {max(0, left):,} of "
                                 f"{settings.playground_tts_chars_per_day:,} characters left. It renews at midnight, "
                                 "India time. For more, call the API with a key.")


async def check_stt(app, customer: Customer, data: bytes) -> float:
    """Seconds of audio in an upload, refused when it is longer than the playground takes or than what is
    left of today's allowance. Decoded here, before anything is transcribed."""
    try:
        seconds = round(pcm_seconds(await ffmpeg_convert(data, 16000, "s16le"), 16000), 2)
    except AudioError:
        raise HTTPException(400, "That audio could not be read. Try another file or record again.") from None
    if seconds > settings.playground_stt_max_seconds:
        raise HTTPException(413, f"The playground transcribes up to {settings.playground_stt_max_seconds:.0f} seconds "
                                 "at a time. For longer audio, call the API with a key.")
    if not customer.free:
        left = settings.playground_stt_seconds_per_day - _used(app, customer)["stt_seconds"]
        if seconds > left:
            raise HTTPException(429, f"That is more than today's playground allowance: {max(0, left):.0f} of "
                                     f"{settings.playground_stt_seconds_per_day:.0f} seconds left. It renews at "
                                     "midnight, India time. For more, call the API with a key.")
    return seconds


@router.get("/v1/playground", include_in_schema=False)
async def playground_allowance(request: Request, customer: Customer = CurrentCustomer):
    """The playground's limits and what is left of them today. Dashboard sessions only."""
    if not customer.playground:
        raise HTTPException(404, "Not Found")
    return allowance(request.app, customer)
