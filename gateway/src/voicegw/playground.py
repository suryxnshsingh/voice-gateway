"""Limits for the dashboard's playground, and demo voices.

The playground calls the public API with a signed in dashboard session instead of an API key
(auth.customer_from). It is there to hear a voice on a few short lines, so a session gets far less
than a key: short texts and recordings, a small daily allowance per account, a few requests a
minute, one at a time, and only the routes in auth.PLAYGROUND_ROUTES. All the numbers are in
config.py (playground_*, demo_*). Keys are unaffected.

A demo voice (Supabase voices.demo, set by an admin) plays only here: a key gets 403
voice_demo_only on every path (public_api._voice_for). In the playground it is free, needs no key
and skips the credit check, so a new account hears it on a balance of 0. Instead it has caps per
voice: demo_tts_chars_per_day, and demo_tts_chars_total since it became a demo, until demo_until.
Its requests are logged with source 'demo', which Supabase prices at 0, and do not count against
the account's playground allowance.

Allowances are counted from this gateway's own request log (db.requests), which every request writes
before its slot is released, so the next request already sees it. Days are India days. An admin
without a key (Customer.free) has no daily playground allowance; demo caps apply to everyone.
"""

import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Request

from .accounts import CodedError
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


def _date(ts: float) -> str:
    d = datetime.fromtimestamp(ts, IST)
    return f"{d.day} {d:%B %Y}"


def limiter(app, customer: Customer):
    """The playground's own limiter (a few a minute, one at a time) for a session; the API's for a key."""
    return app.state.playground_limiter if customer.playground else app.state.limiter


def _used(app, customer: Customer) -> dict[str, float]:
    return app.state.db.playground_used(customer.id, day_start())


def needs_key(customer: Customer) -> None:
    """Everything but a demo voice is billed, to the key the playground names (an admin needs none)."""
    if customer.key_id is None and not customer.free:
        raise HTTPException(400, "Choose one of your API keys to use the playground.")


def check_tts_text(text: str) -> None:
    if len(text) > settings.playground_tts_max_chars:
        raise HTTPException(413, f"The playground speaks up to {settings.playground_tts_max_chars} characters at a time. "
                                 "For longer text, call the API with a key.")


def check_tts(app, customer: Customer, voice: dict, characters: int) -> str:
    """Refuses a playground request past its limits. Returns the source to log it under: 'demo' for a
    demo voice (free, its own caps), else 'playground' (billed, the account's daily allowance)."""
    if voice.get("demo"):
        check_demo(app, customer, voice, characters)
        return "demo"
    needs_key(customer)
    check_tts_allowance(app, customer, characters)
    return "playground"


def check_tts_allowance(app, customer: Customer, characters: int) -> None:
    if customer.free:
        return
    left = settings.playground_tts_chars_per_day - _used(app, customer)["characters"]
    if characters > left:
        raise HTTPException(429, f"That is more than today's playground allowance: {max(0, left):,} of "
                                 f"{settings.playground_tts_chars_per_day:,} characters left. It renews at midnight, "
                                 "India time. For more, call the API with a key.")


def demo_left(app, customer: Customer, voice: dict) -> dict:
    """What is left of a demo voice: today, and of the whole demo."""
    db = app.state.db
    today = db.playground_used(customer.id, day_start(), "demo", voice["id"])
    total = db.playground_used(customer.id, voice.get("demo_since") or 0, "demo", voice["id"])
    until = voice.get("demo_until")
    return {
        "voice_id": voice["id"],
        "demo_until": until,
        "ended": until is not None and until <= time.time(),
        "characters_left_today": max(0, settings.demo_tts_chars_per_day - today["characters"]),
        "characters_left_total": max(0, settings.demo_tts_chars_total - total["characters"]),
        "last_played_at": total["last_at"],
    }


def check_demo(app, customer: Customer, voice: dict, characters: int) -> None:
    left = demo_left(app, customer, voice)
    if left["ended"]:
        raise CodedError(403, "voice_demo_ended", f"This demo ended on {_date(left['demo_until'])}. "
                         "Unlock the voice to keep using it.")
    if characters > left["characters_left_total"]:
        raise CodedError(429, "voice_demo_used_up", f"That is more than is left of this demo: "
                         f"{left['characters_left_total']:,} of {settings.demo_tts_chars_total:,} characters. "
                         "Unlock the voice to keep using it.")
    if characters > left["characters_left_today"]:
        raise CodedError(429, "voice_demo_daily", f"That is more than today's demo allowance for this voice: "
                         f"{left['characters_left_today']:,} of {settings.demo_tts_chars_per_day:,} characters left. "
                         "It renews at midnight, India time.")


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


async def allowance(app, customer: Customer) -> dict:
    """What a session may still do today, as the dashboard shows it. None means no daily limit."""
    used = None if customer.free else _used(app, customer)
    demos = [v for v in await app.state.accounts.voices(customer) if v.get("demo")]
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
        "demo": {
            "characters_per_day": settings.demo_tts_chars_per_day,
            "characters_total": settings.demo_tts_chars_total,
            "voices": [demo_left(app, customer, v) for v in demos],
        },
        "resets_at": day_start() + 86400,
    }


@router.get("/v1/playground", include_in_schema=False)
async def playground_allowance(request: Request, customer: Customer = CurrentCustomer):
    """The playground's limits and what is left of them today. Dashboard sessions only."""
    if not customer.playground:
        raise HTTPException(404, "Not Found")
    return await allowance(request.app, customer)


@router.post("/v1/playground/refresh", include_in_schema=False)
async def refresh_voices(request: Request, customer: Customer = CurrentCustomer):
    """After the dashboard unlocks a voice: read the account's voices again now, not when the cache expires,
    so the voice works with a key at once."""
    if not customer.playground:
        raise HTTPException(404, "Not Found")
    request.app.state.playground_limiter.check_rate(customer.id)
    forget = getattr(request.app.state.accounts, "forget_voices", None)
    if forget:
        forget(customer.id)
    return await allowance(request.app, customer)
