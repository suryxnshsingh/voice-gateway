"""Speech-to-text benchmark on English, Hindi and Hinglish.

Streaming models get the audio at real-time pace, with turns cut by the gateway's own VAD logic,
so the latency numbers are what a caller would see. Batch and local models get whole clips.
Accuracy is the voice-clone-poc's script-agnostic consonant-skeleton error: "Meselson" and
"मेसलसन" match, so a model isn't punished for writing an English word in Devanagari or back.

    cd bench && uv run python stt_bench.py            # everything available
    uv run python stt_bench.py --only gpt-live-transcribe,whisper-turbo-local
"""

import argparse
import asyncio
import io
import json
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from indic_transliteration import sanscript
from openai import AsyncOpenAI

from voicegw.config import Settings
from voicegw.frontend import AudioFrontend, VadParams
from voicegw.providers.openai_stt import OpenAIRealtimeSTT
from voicegw.providers.soniox import SonioxStream, SttError

ROOT = Path(__file__).resolve().parent
POC = ROOT.parent.parent / "voice-clone-poc"
settings = Settings(_env_file=ROOT.parent / "gateway" / ".env")
HINTS = ["en", "hi"]


@dataclass
class Clip:
    id: str
    lang: str          # en | hi | hinglish
    path: Path
    reference: str
    exact: bool        # False: the reference is itself a Whisper transcript (real recordings)


def clips() -> list[Clip]:
    out = []
    # (script, TTS render used as the audio, language group). Renders are Indian voice clones from
    # the voice-clone-poc, so the reference text is exactly what was spoken.
    for sid, render, lang in [
        ("01_hi_ncert_dna", "sarvam", "hi"), ("02_hi_ncert_numbers", "inworld-tts-2", "hi"),
        ("03_hinglish_devanagari", "inworld-tts-2", "hinglish"), ("04_hinglish_roman", "sarvam", "hinglish"),
        ("05_en_explainer", "inworld-tts-2", "en"), ("08_hinglish_terms_dense", "sarvam", "hinglish"),
        ("09_hinglish_english_clauses", "inworld-tts-2", "hinglish"), ("10_english_in_devanagari", "sarvam", "hinglish"),
    ]:
        out.append(Clip(f"{sid}", lang, POC / "out/runs/ggsir" / render / f"{sid}.mp3",
                        (POC / "texts" / f"{sid}.txt").read_text().strip(), True))
    for ref in ("ritesh", "ggsir"):  # real teachers; ggsir is phone-quality (band-limited at ~4 kHz)
        out.append(Clip(f"real_{ref}_30s", "hinglish", POC / "refs" / ref / "ref_30s.wav",
                        (POC / "refs" / ref / "ref_30s.txt").read_text().strip(), False))
    return out


def pcm16k(path: Path) -> bytes:
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                          capture_output=True, check=True).stdout


def wav16k(pcm: bytes) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.frombuffer(pcm, dtype="<i2"), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# --- scoring (same as voice-clone-poc/voicebench/intelligibility.py) ---------------------------

def skeleton(text: str) -> str:
    roman = sanscript.transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    return re.sub(r"(.)\1+", r"\1", re.sub(r"[^a-z0-9]|[aeiouh]", "", roman))


def edits(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(reference: str, heard: str) -> dict:
    s, h = skeleton(reference), skeleton(heard)
    return {"skel_error": round(edits(s, h) / max(len(s), 1), 3), "coverage": round(len(h) / max(len(s), 1), 3)}


# --- streaming: the gateway's adapter + VAD endpointing, at real-time pace ----------------------

async def run_streaming(stream, pcm: bytes, frame_ms: int = 40) -> dict:
    fe = AudioFrontend(VadParams(), noise_suppression=True)
    t0 = time.monotonic()
    first_speech = first_text = None
    commits: list[float] = []   # wall times we committed
    finals: list[tuple[float, str]] = []
    endpoint_task = None
    speaking = False
    turn_silence = 0.8 - 0.5   # gateway default turn_silence_ms - vad_stop_ms

    async def endpoint_after_silence():
        await asyncio.sleep(turn_silence)
        if not speaking:
            commits.append(time.monotonic())
            await stream.finalize()

    async def receive():
        nonlocal first_text
        async for up in stream.updates():
            now = time.monotonic()
            if first_text is None and (up.partial or up.finals):
                first_text = now
            if up.finals:
                finals.append((now, "".join(t.text for t in up.finals)))

    receiver = asyncio.create_task(receive())
    step = 16000 * 2 * frame_ms // 1000
    padded = pcm + bytes(16000 * 2 * 2)  # 2 s of trailing silence so the last turn ends
    for i in range(0, len(padded), step):
        if receiver.done():
            break  # the provider failed; don't stream the rest
        frame = padded[i : i + step]
        target = t0 + i / (16000 * 2)
        await asyncio.sleep(max(0, target - time.monotonic()))
        try:
            await stream.send_audio(frame)
        except Exception:
            await asyncio.sleep(0.2)  # the provider hung up; its reason arrives on the receive side
            if receiver.done() and receiver.exception():
                raise receiver.exception()
            raise
        res = fe.process(frame)
        for ev in res.events:
            if ev.kind == "start":
                speaking = True
                first_speech = first_speech or time.monotonic()
                if endpoint_task:
                    endpoint_task.cancel()
            elif ev.kind == "stop":
                speaking = False
                if stream.needs_endpointing:
                    endpoint_task = asyncio.create_task(endpoint_after_silence())
    if not stream.needs_endpointing:
        await stream.end_audio()
    # wait for the last final
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and (len(finals) < len(commits) if stream.needs_endpointing else not receiver.done()):
        await asyncio.sleep(0.05)
    if receiver.done() and receiver.exception():
        await stream.close()
        fe.close()
        raise receiver.exception()
    receiver.cancel()
    await stream.close()
    fe.close()
    end_of_audio = t0 + len(pcm) / (16000 * 2)
    final_lags = [f[0] - c for c, f in zip(commits, finals)]
    return {
        # Soniox finals are sub-word pieces that carry their own spaces; OpenAI finals are whole turns.
        "text": (" " if stream.needs_endpointing else "").join(t for _, t in finals).strip(),
        "first_text_ms": round((first_text - (first_speech or t0)) * 1000) if first_text else None,
        # From the end of the clip's speech to its last final text: what end-of-turn feels like.
        "last_final_after_audio_end_ms": round((finals[-1][0] - end_of_audio) * 1000) if finals else None,
        "final_after_commit_ms": round(statistics.median(final_lags) * 1000) if final_lags else None,
        "turns": len(finals),
    }


async def openai_streaming(model: str, pcm: bytes, **kw) -> dict:
    stream = await OpenAIRealtimeSTT.connect(api_key=settings.openai_key, model=model, language_hints=HINTS, **kw)
    return await run_streaming(stream, pcm)


async def soniox_streaming(pcm: bytes) -> dict:
    stream = await SonioxStream.connect(api_key=settings.soniox_api_key, url=settings.soniox_ws_url,
                                        model=settings.soniox_model, language_hints=HINTS)
    return await run_streaming(stream, pcm)


# --- batch ------------------------------------------------------------------------------------

async def openai_batch(model: str, pcm: bytes) -> dict:
    client = AsyncOpenAI(api_key=settings.openai_key)
    t0 = time.monotonic()
    r = await client.audio.transcriptions.create(model=model, file=("clip.wav", wav16k(pcm)))
    return {"text": r.text, "wall_ms": round((time.monotonic() - t0) * 1000)}


_mlx = None


def whisper_turbo_local(pcm: bytes) -> dict:
    import mlx_whisper

    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    t0 = time.monotonic()
    r = mlx_whisper.transcribe(x, path_or_hf_repo="mlx-community/whisper-large-v3-turbo", verbose=None)
    return {"text": r["text"], "wall_ms": round((time.monotonic() - t0) * 1000), "language": r.get("language")}


_hinglish = None


def hinglish_whisper_local(pcm: bytes) -> dict:
    global _hinglish
    import torch
    from transformers import pipeline

    if _hinglish is None:
        _hinglish = pipeline("automatic-speech-recognition", model="Oriserve/Whisper-Hindi2Hinglish-Prime",
                             device="mps", torch_dtype=torch.float16, chunk_length_s=30)
    x = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
    t0 = time.monotonic()
    r = _hinglish({"raw": x, "sampling_rate": 16000}, generate_kwargs={"task": "transcribe"}, return_timestamps=True)
    return {"text": r["text"], "wall_ms": round((time.monotonic() - t0) * 1000)}


PROVIDERS = {
    # name: (kind, runner)
    "soniox-stt-rt-v3": ("stream", lambda pcm: soniox_streaming(pcm)),
    "gpt-live-transcribe": ("stream", lambda pcm: openai_streaming("gpt-live-transcribe", pcm)),
    "gpt-live-transcribe-minimal": ("stream", lambda pcm: openai_streaming("gpt-live-transcribe", pcm, delay="minimal",
                                                                           keywords=settings.stt_keywords)),
    "gpt-realtime-whisper": ("stream", lambda pcm: openai_streaming("gpt-realtime-whisper", pcm)),
    "gpt-transcribe": ("batch", lambda pcm: openai_batch("gpt-transcribe", pcm)),
    "gpt-4o-transcribe": ("batch", lambda pcm: openai_batch("gpt-4o-transcribe", pcm)),
    "gpt-4o-mini-transcribe": ("batch", lambda pcm: openai_batch("gpt-4o-mini-transcribe", pcm)),
    "whisper-1": ("batch", lambda pcm: openai_batch("whisper-1", pcm)),
    "whisper-turbo-local": ("local", whisper_turbo_local),
    "hinglish-whisper-local": ("local", hinglish_whisper_local),
}


async def run_one(name: str, clip: Clip, pcm: bytes) -> dict:
    kind, runner = PROVIDERS[name]
    base = {"provider": name, "kind": kind, "clip": clip.id, "lang": clip.lang, "exact_ref": clip.exact,
            "audio_s": round(len(pcm) / 32000, 1)}
    try:
        if kind == "local":
            res = await asyncio.to_thread(runner, pcm)  # one at a time: they share the GPU
        else:
            res = await runner(pcm)
    except (SttError, Exception) as e:
        return {**base, "failure": f"{type(e).__name__}: {str(e)[:160]}"}
    return {**base, **res, **score(clip.reference, res["text"])}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated provider names")
    ap.add_argument("--out", default=str(ROOT / "results" / "stt.json"))
    ap.add_argument("--report", action="store_true", help="re-print the report from --out without running")
    ap.add_argument("--merge", action="store_true", help="keep other providers' rows already in --out")
    args = ap.parse_args()
    names = args.only.split(",") if args.only else list(PROVIDERS)
    if args.report:
        report(json.loads(Path(args.out).read_text()), names)
        return
    data = [(c, pcm16k(c.path)) for c in clips()]
    print(f"{len(data)} clips, {sum(len(p) for _, p in data) / 32000 / 60:.1f} min of audio")

    rows = []
    # Cloud providers run concurrently (real-time streaming takes as long as the audio); local
    # models run one clip at a time.
    # Bounded concurrency: dozens of simultaneous sessions slow the provider down and the latency
    # numbers stop meaning anything (seen: finals >15 s late, batch at 1.3x real time).
    sems = {"stream": asyncio.Semaphore(2), "batch": asyncio.Semaphore(1)}

    async def bounded(n, c, p):
        async with sems[PROVIDERS[n][0]]:
            row = await run_one(n, c, p)
            print(f"  {n} {c.id}: {row.get('skel_error', row.get('failure'))}")
            return row

    cloud = [bounded(n, c, p) for n in names if PROVIDERS[n][0] != "local" for c, p in data]
    local = [n for n in names if PROVIDERS[n][0] == "local"]

    async def run_local():
        out = []
        for n in local:
            for c, p in data:
                out.append(await run_one(n, c, p))
                print(f"  {n} {c.id}: done")
        return out

    cloud_rows, local_rows = await asyncio.gather(asyncio.gather(*cloud), run_local())
    rows = list(cloud_rows) + local_rows
    if args.merge and Path(args.out).exists():
        rows = [r for r in json.loads(Path(args.out).read_text()) if r["provider"] not in names] + rows
        names = list(dict.fromkeys([r["provider"] for r in rows]))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    report(rows, names)


def report(rows: list[dict], names: list[str]) -> None:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(statistics.mean(xs), 3) if xs else None

    print(f"\n{'provider':24s} {'en':>6s} {'hi':>6s} {'hinglish':>9s} {'all':>6s}   latency")
    for n in names:
        rs = [r for r in rows if r["provider"] == n]
        good = [r for r in rs if "failure" not in r]
        if not good:
            print(f"{n:24s} failed: {rs[0]['failure']}")
            continue
        by = lambda lang: mean([r["skel_error"] for r in good if r["lang"] == lang and r["exact_ref"]])
        allx = mean([r["skel_error"] for r in good if r["exact_ref"]])
        if PROVIDERS[n][0] == "stream":
            lat = (f"first words {mean([r['first_text_ms'] for r in good])} ms after speech starts; "
                   f"final {mean([r['final_after_commit_ms'] for r in good])} ms after commit")
        else:
            rtf = mean([r["wall_ms"] / 1000 / r["audio_s"] for r in good])
            lat = f"{mean([r['wall_ms'] for r in good])} ms per clip (RTF {rtf})"
        errs = len(rs) - len(good)
        print(f"{n:24s} {by('en')!s:>6s} {by('hi')!s:>6s} {by('hinglish')!s:>9s} {allx!s:>6s}   {lat}"
              + (f"  [{errs} failed]" if errs else ""))
    print("\nerror = consonant-skeleton edit rate vs the script (lower is better; ~0.25+ means dropped or garbled words).")
    print("Scored on the 8 clips with exact scripts. The two real teacher recordings have no trustworthy reference")
    print("(the Whisper-made one covers half the clip), so they are shown side by side instead:")
    for clip in sorted({r["clip"] for r in rows if not r["exact_ref"]}):
        print(f"\n## {clip}")
        for n in names:
            r = next((r for r in rows if r["provider"] == n and r["clip"] == clip), None)
            if r and "text" in r:
                print(f"  {n:24s} {r['text'][:150]!r}")


if __name__ == "__main__":
    asyncio.run(main())
