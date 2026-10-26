"""Speech-to-text provider chain: try providers in order, skip ones that recently failed."""

import logging
import time

from ..config import Settings
from . import openai_stt, soniox
from .soniox import SttError

log = logging.getLogger(__name__)

# provider -> monotonic time until which it is skipped. Process-wide, so one failure spares every
# other session the same failed connect.
_down_until: dict[str, float] = {}


def _cooldown(err: Exception) -> float:
    status = getattr(err, "status", None)
    if status in (401, 402, 403):
        return 600  # bad key or out of credit: won't fix itself in seconds
    if status == 429:
        return 60
    return 20


def mark_down(provider: str, err: Exception) -> None:
    seconds = _cooldown(err)
    _down_until[provider] = time.monotonic() + seconds
    log.warning("stt provider %s unavailable for %ds: %s", provider, seconds, err)


def available(settings: Settings) -> list[str]:
    now = time.monotonic()
    return [p for p in settings.stt_providers if _down_until.get(p, 0) <= now and _configured(p, settings)]


def _configured(provider: str, s: Settings) -> bool:
    return bool(s.soniox_api_key) if provider == "soniox" else bool(s.openai_key)


async def _connect(provider: str, s: Settings, language_hints: list[str]):
    if provider == "soniox":
        return await soniox.SonioxStream.connect(api_key=s.soniox_api_key, url=s.soniox_ws_url, model=s.soniox_model,
                                                 language_hints=language_hints)
    if provider == "openai":
        return await openai_stt.OpenAIRealtimeSTT.connect(api_key=s.openai_key, model=s.openai_stt_model,
                                                          language_hints=language_hints, delay=s.openai_stt_delay,
                                                          keywords=s.stt_keywords)
    raise SttError(f"unknown stt provider {provider}")


async def connect(s: Settings, language_hints: list[str]):
    errors = []
    for provider in available(s):
        try:
            return await _connect(provider, s, language_hints)
        except Exception as e:
            mark_down(provider, e)
            errors.append(f"{provider}: {e}")
    raise SttError("no speech-to-text provider available" + (f" ({'; '.join(errors)})" if errors else ""))


async def transcribe(pcm16k: bytes, s: Settings, language_hints: list[str]) -> tuple[str, str]:
    """Whole clip -> (transcript, provider used)."""
    errors = []
    for provider in available(s):
        try:
            if provider == "soniox":
                text = await soniox.transcribe(pcm16k, api_key=s.soniox_api_key, url=s.soniox_ws_url,
                                               model=s.soniox_model, language_hints=language_hints)
            else:
                text = await openai_stt.transcribe(pcm16k, api_key=s.openai_key, model=s.openai_stt_model,
                                                   language_hints=language_hints, keywords=s.stt_keywords)
            return text, provider
        except Exception as e:
            mark_down(provider, e)
            errors.append(f"{provider}: {e}")
    raise SttError("no speech-to-text provider available" + (f" ({'; '.join(errors)})" if errors else ""))
