"""OpenAI-compatible chat completions (OpenAI, OpenRouter, Gemini's compat endpoint)."""

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import openai
from openai import AsyncOpenAI
from openai._constants import DEFAULT_CONNECTION_LIMITS


def _http_client():
    # The SDK drops idle connections after 5 s, so most voice turns paid a fresh TCP+TLS handshake
    # (130-650 ms measured from India). Keep them for 5 minutes. Built from the SDK's own limits type
    # so this survives the SDK switching HTTP libraries (it moved httpx -> httpx2).
    limits = type(DEFAULT_CONNECTION_LIMITS)(max_connections=100, max_keepalive_connections=20, keepalive_expiry=300)
    return openai.DefaultAsyncHttpxClient(limits=limits)


@dataclass
class LlmUsage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLM:
    def __init__(self, api_key: str, base_url: str | None, service_tier: str = ""):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=_http_client()) if api_key else None
        self._service_tier = service_tier  # "priority": ~200 ms sooner first sentence at 2x price (bench, Oct 2026)
        self._last_warm = 0.0

    async def warm(self, model: str) -> None:
        """Open a connection before it's needed (called when the user starts talking). Cheap GET; errors ignored."""
        if self._client is None or time.monotonic() - self._last_warm < 20:
            return
        self._last_warm = time.monotonic()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._client.models.retrieve(model), 5)

    @property
    def available(self) -> bool:
        return self._client is not None

    async def stream(self, model: str, messages: list[dict], max_tokens: int, usage: LlmUsage,
                     reasoning_effort: str | None = None) -> AsyncIterator[str]:
        extra = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
        if self._service_tier:
            extra["service_tier"] = self._service_tier
        stream = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            max_completion_tokens=max_tokens,
            **extra,
        )
        try:
            async for chunk in stream:
                if chunk.usage:
                    usage.input_tokens = chunk.usage.prompt_tokens
                    usage.output_tokens = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        finally:
            await stream.close()

    async def json(self, model: str, messages: list[dict], max_tokens: int, usage: LlmUsage) -> str:
        resp = await self._client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens,
        )
        if resp.usage:
            usage.input_tokens = resp.usage.prompt_tokens
            usage.output_tokens = resp.usage.completion_tokens
        return resp.choices[0].message.content or "{}"
