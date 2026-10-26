"""Batch transcription for the public STT endpoint: the file goes to the configured engine as-is.

Errors are mapped to neutral messages here, so nothing about the engine reaches the customer.
"""

import logging
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI

log = logging.getLogger(__name__)


class TranscribeError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass
class Transcript:
    text: str
    seconds: float | None  # billed audio duration when the engine reports it


class Transcriber:
    def __init__(self, api_key: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key, timeout=120, max_retries=1) if api_key else None
        self._model = model

    @property
    def available(self) -> bool:
        return self._client is not None

    async def transcribe(self, data: bytes, filename: str, language: str | None, prompt: str | None) -> Transcript:
        kwargs = {}
        if language:
            kwargs["language"] = language
        if prompt:
            kwargs["prompt"] = prompt
        try:
            r = await self._client.audio.transcriptions.create(model=self._model, file=(filename, data), **kwargs)
        except openai.BadRequestError as e:
            log.warning("transcription rejected: %s", e)
            raise TranscribeError(400, "The audio could not be processed. Check the file format and that it contains speech.") from e
        except openai.RateLimitError as e:
            log.warning("transcription rate limited: %s", e)
            raise TranscribeError(429, "Transcription is busy right now. Please retry shortly.") from e
        except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as e:
            log.error("transcription unavailable: %s", e)
            raise TranscribeError(503, "Transcription is temporarily unavailable. Please retry.") from e
        except openai.APIError as e:
            log.error("transcription failed: %s", e)
            raise TranscribeError(502, "Transcription failed. Please retry.") from e
        usage = getattr(r, "usage", None)
        seconds = getattr(usage, "seconds", None) if usage is not None and getattr(usage, "type", "") == "duration" else None
        return Transcript(text=r.text, seconds=seconds)
