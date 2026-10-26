"""Tone from audio + transcript.

emotion2vec hears how something was said but not what; it reads most calm speech as neutral and
has no notion of "confused". The fusion step gives a small LLM both signals and asks for one label
from a taxonomy that fits tutoring and support conversations.
"""

import json

from .providers.llm import LLM, LlmUsage

TONES = ["neutral", "happy", "excited", "confident", "confused", "frustrated", "bored", "anxious", "sad", "angry"]

_SYSTEM = (
    "You label the emotional tone of one spoken utterance in a live voice conversation.\n"
    "You get the transcript and acoustic emotion scores from a speech-emotion model. The acoustic model "
    "hears pitch, energy and pace but not the words; it reads calm speech as neutral and cannot detect "
    "confusion. The words alone miss sarcasm and irritation. Weigh both.\n"
    "Asking a question is not confusion by itself: label confused only when they say they don't understand, "
    "repeat a question, or sound lost. A plain question asked calmly is neutral.\n"
    f"Reply with JSON only: {{\"tone\": one of {TONES}, \"confidence\": 0..1, \"reason\": \"<10 words\"}}"
)


def acoustic_summary(acoustic: dict | None) -> str:
    if not acoustic:
        return "unavailable"
    top = sorted(acoustic["scores"].items(), key=lambda kv: -kv[1])[:3]
    return ", ".join(f"{k} {v:.2f}" for k, v in top)


async def fuse(llm: LLM, model: str, transcript: str, acoustic: dict | None, usage: LlmUsage) -> dict:
    raw = await llm.json(
        model,
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Transcript: {transcript}\nAcoustic: {acoustic_summary(acoustic)}"},
        ],
        max_tokens=60,
        usage=usage,
    )
    data = json.loads(raw)
    tone = data.get("tone") if data.get("tone") in TONES else "neutral"
    return {"label": tone, "confidence": float(data.get("confidence", 0.5)), "reason": str(data.get("reason", ""))[:120]}
