"""Reply-LLM benchmark for the voice loop.

What matters in a voice reply is how soon the first speakable sentence exists (TTS can't start
before it), then whether the reply follows the voice rules: short, no markdown, same language.

    cd gateway && uv run python ../bench/llm_bench.py --runs 3
"""

import argparse
import asyncio
import json
import re
import statistics
import time
from pathlib import Path

from openai import AsyncOpenAI

from voicegw.config import settings
from voicegw.session import VOICE_RULES
from voicegw.text import SentenceSplitter

PERSONA = "You are Instant Guru, a friendly Class 12 biology tutor for Indian students."

CASES = [
    ("en_question", "en", [("user", "What is photosynthesis? Explain it simply.")]),
    ("hinglish", "hinglish", [("user", "Sir mitochondria ko powerhouse of the cell kyun bolte hain?")]),
    ("hindi", "hi", [("user", "डीएनए और आरएनए में क्या अंतर है?")]),
    ("confused", "en", [("user", "wait I didn't get the Calvin cycle part at all\n[tone: confused]")]),
    ("followup", "en", [
        ("user", "What is osmosis?"),
        ("assistant", "Osmosis is the movement of water across a membrane from where there is more water to where there is less."),
        ("user", "Okay, and how is that different from diffusion?"),
    ]),
    ("hindi_short", "hi", [("user", "एंजाइम क्या होते हैं? एक लाइन में बताइए।")]),
]

MODELS = [
    ("gpt-5.4-mini", {}),
    ("gpt-6-luna", {}),
    ("gpt-6-luna", {"reasoning_effort": "none"}),
    ("gpt-6-luna", {"reasoning_effort": "low"}),
    ("gpt-6-sol", {"reasoning_effort": "none"}),
    ("gpt-5.6-luna", {"reasoning_effort": "none"}),
    ("gpt-4.1-mini", {}),
]

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
MARKDOWN = re.compile(r"(\*\*|__|^#|^\s*[-•]\s|^\s*\d+\.\s|`)", re.M)


def label(model, kw):
    tags = [kw[k] for k in ("reasoning_effort", "service_tier") if k in kw]
    return model + (f" ({', '.join(tags)})" if tags else "")


def checks(text: str, lang: str) -> dict:
    sentences = [s for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s]
    deva = len(DEVANAGARI.findall(text)) / max(1, len(re.findall(r"\w", text)))
    lang_ok = deva > 0.5 if lang == "hi" else (deva < 0.1 if lang in ("en", "hinglish") else True)
    return {"sentences": len(sentences), "chars": len(text), "markdown": bool(MARKDOWN.search(text)),
            "lang_ok": lang_ok, "rules_ok": len(sentences) <= 4 and not MARKDOWN.search(text) and lang_ok}


async def one(client, model, kw, case, sem):
    name, lang, turns = case
    msgs = [{"role": "system", "content": VOICE_RULES + "\n\n" + PERSONA}] + [{"role": r, "content": c} for r, c in turns]
    async with sem:
        t0 = time.monotonic()
        first_tok = first_sentence = None
        text, usage = "", None
        splitter = SentenceSplitter()
        try:
            stream = await client.chat.completions.create(model=model, messages=msgs, stream=True,
                                                          stream_options={"include_usage": True},
                                                          max_completion_tokens=settings.llm_max_tokens, **kw)
            async for ch in stream:
                if ch.usage:
                    usage = ch.usage
                if ch.choices and ch.choices[0].delta.content:
                    d = ch.choices[0].delta.content
                    now = time.monotonic() - t0
                    first_tok = first_tok if first_tok is not None else now
                    text += d
                    if first_sentence is None and splitter.feed(d):
                        first_sentence = now
            total = time.monotonic() - t0
            if first_sentence is None and text:
                first_sentence = total
        except Exception as e:
            return {"model": label(model, kw), "case": name, "error": str(e)[:200]}
    reasoning = getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None) if usage else None
    return {"model": label(model, kw), "case": name, "first_token_ms": round(first_tok * 1000) if first_tok else None,
            "first_sentence_ms": round(first_sentence * 1000) if first_sentence else None, "total_ms": round(total * 1000),
            "output_tokens": usage.completion_tokens if usage else None, "reasoning_tokens": reasoning,
            "text": text, **checks(text, lang)}


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--set", default="default", choices=["default", "tiers"])
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--out", default=str(Path(__file__).parent / "results" / "llm.json"))
    args = ap.parse_args()
    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    sem = asyncio.Semaphore(args.concurrency)
    global MODELS
    if args.set == "tiers":
        MODELS = [("gpt-5.4-mini", {}), ("gpt-5.4-mini", {"service_tier": "priority"}),
                  ("gpt-6-luna", {"reasoning_effort": "none"}),
                  ("gpt-6-luna", {"reasoning_effort": "none", "service_tier": "priority"})]
    # Interleave models so network weather hits every model alike.
    jobs = [one(client, m, kw, c, sem) for _ in range(args.runs) for c in CASES for m, kw in MODELS]
    rows = await asyncio.gather(*jobs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))

    print(f"{'model':26s} {'1st token p50/p90':>18s} {'1st sentence p50/p90':>21s} {'total p50':>10s} {'chars':>6s} {'rules':>6s} {'err':>4s}")
    for m, kw in MODELS:
        name = label(m, kw)
        ok = [r for r in rows if r["model"] == name and "error" not in r]
        err = sum(1 for r in rows if r["model"] == name and "error" in r)
        if not ok:
            print(f"{name:26s} all failed: {next(r['error'] for r in rows if r['model'] == name)}")
            continue
        ft = [r["first_token_ms"] for r in ok if r["first_token_ms"]]
        fs = [r["first_sentence_ms"] for r in ok if r["first_sentence_ms"]]
        print(f"{name:26s} {pct(ft,.5):>8} / {pct(ft,.9):<8} {pct(fs,.5):>10} / {pct(fs,.9):<9} {pct([r['total_ms'] for r in ok],.5):>10}"
              f" {round(statistics.mean(r['chars'] for r in ok)):>6} {sum(r['rules_ok'] for r in ok):>3}/{len(ok):<2} {err:>4}")
    print("\nsamples (run 1):")
    for c in CASES:
        print(f"\n## {c[0]}: {c[2][-1][1][:70]}")
        for m, kw in MODELS:
            r = next((r for r in rows if r["model"] == label(m, kw) and r["case"] == c[0] and "text" in r), None)
            if r:
                print(f"  {label(m, kw):26s} {r['text'][:160]!r}")


if __name__ == "__main__":
    asyncio.run(main())
