"""Build a VoxCPM2 fine-tuning dataset for one speaker from long recordings.

    cd bench && uv run python ../training/prepare_dataset.py SOURCE.wav [SOURCE2.wav ...] \
        --reference ../gateway/data/voices/<voice_id>/reference.wav --out ../training/datasets/ritesh

Steps:
1. Silero VAD finds speech; phrases are packed into 3-12 s clips that start and end on pauses.
2. Each clip is checked against the reference with ECAPA speaker similarity (via the worker), so
   other voices (students, a co-host) are dropped.
3. gpt-transcribe writes the transcript (the best Hindi accuracy in bench/stt_bench.py).
4. A small LLM rewrites English words into Latin script, Hindi stays Devanagari: the same mixed
   script the gateway's LLM speaks in. The rewrite is rejected if its consonant skeleton drifts from
   the original, i.e. if it changed words rather than script.
5. Writes train.jsonl / val.jsonl ({"audio", "text", "duration"}) plus a report.
"""

import argparse
import asyncio
import io
import json
import re
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import soxr
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "bench"))
from stt_bench import score  # noqa: E402  (script-agnostic skeleton error)
from voicegw.config import Settings  # noqa: E402
from voicegw.frontend import AudioFrontend, VadParams, float_to_pcm16  # noqa: E402

settings = Settings(_env_file=ROOT / "gateway" / ".env")
SR = 48000

REWRITE = """You fix the script of a Hindi/Hinglish transcript for a text-to-speech dataset.
Write every English word (and English acronym, brand or technical term) in English letters; keep
every Hindi word in Devanagari. Examples: "वीडियो क्लिप्स" -> "video clips", "बैच" -> "batch",
"परसेंट" -> "percent", "एनसीईआरटी" -> "NCERT". Numbers stay as written.
The brand name is "Arivihan" (speech recognition often writes it as रविहान or अरविहान): write it as Arivihan.
Apart from that, do not add, remove, reorder, translate or correct any word. Keep the punctuation.
Reply with the rewritten transcript only."""

# Names the transcriber should expect. Passed as the transcription prompt.
VOCAB = "Arivihan, Ritesh Sir, MP Board, Class 11, Class 12, Unnati Batch, NEET, JEE, NCERT, app"


def segments(x16: np.ndarray) -> list[tuple[float, float]]:
    # Short stop_ms: fluent speakers rarely pause long, so cut points have to be word gaps.
    fe = AudioFrontend(VadParams(threshold=0.5, start_ms=96, stop_ms=120), noise_suppression=False)
    pcm = float_to_pcm16(x16)
    out, start = [], None
    for i in range(0, len(pcm), 1280):
        for ev in fe.process(pcm[i : i + 1280]).events:
            now = (i + 1280) / 32000
            if ev.kind == "start":
                start = now - 0.15
            else:
                out.append((max(start, 0.0), now - 0.2))
    fe.close()
    return out


def split_long(x16: np.ndarray, a: float, b: float, max_s: float) -> list[tuple[float, float]]:
    """A stretch with no pause longer than max_s: cut at its quietest 30 ms between 5 s and max_s."""
    out = []
    while b - a > max_s:
        lo, hi = int((a + 5) * 16000), int((a + max_s) * 16000)
        frames = x16[lo:hi][: (hi - lo) // 480 * 480].reshape(-1, 480)
        cut = a + 5 + int(np.argmin((frames ** 2).mean(axis=1))) * 0.03 + 0.015
        out.append((a, cut))
        a = cut
    out.append((a, b))
    return out


def pack(segs: list[tuple[float, float]], x16: np.ndarray, min_s=3.0, max_s=12.0) -> list[tuple[float, float]]:
    """Speech segments -> clips that start and end on a gap and stay within [min_s, max_s]."""
    phrases = [p for a, b in segs for p in split_long(x16, a, b, max_s)]
    clips, cur = [], None
    for a, b in phrases:
        if cur is None:
            cur = (a, b)
        elif b - cur[0] <= max_s:
            cur = (cur[0], b)
        else:
            if cur[1] - cur[0] >= min_s:
                clips.append(cur)
            cur = (a, b)
    if cur and cur[1] - cur[0] >= min_s:
        clips.append(cur)
    return clips


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-speaker", type=float, default=0.55)
    ap.add_argument("--val-fraction", type=float, default=0.04)
    ap.add_argument("--concurrency", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    ref_wav = Path(args.reference).read_bytes()
    oai = AsyncOpenAI(api_key=settings.openai_key)
    worker = httpx.AsyncClient(base_url=settings.worker_url, timeout=120,
                               headers={"x-worker-token": settings.worker_token} if settings.worker_token else {})
    sem = asyncio.Semaphore(args.concurrency)
    rows, dropped = [], {"speaker": 0, "quiet": 0, "clipping": 0, "rewrite_drift": 0, "empty": 0}

    async def handle(cid: str, clip: np.ndarray):
        async with sem:
            data = wav_bytes(clip, SR)
            if dbfs(clip) < -40:
                dropped["quiet"] += 1
                return
            if np.mean(np.abs(clip) > 0.99) > 0.001:
                dropped["clipping"] += 1
                return
            r = await worker.post("/speaker/similarity", files={"a": ("a.wav", ref_wav), "b": ("b.wav", data)})
            sim = r.json()["score"]
            if sim < args.min_speaker:
                dropped["speaker"] += 1
                return
            tr = await oai.audio.transcriptions.create(model="gpt-transcribe", file=(f"{cid}.wav", data), prompt=VOCAB)
            raw = tr.text.strip()
            if not raw:
                dropped["empty"] += 1
                return
            resp = await oai.chat.completions.create(
                model="gpt-5.4-mini", messages=[{"role": "system", "content": REWRITE}, {"role": "user", "content": raw}])
            text = resp.choices[0].message.content.strip()
            drift = score(raw, text)["skel_error"]
            if drift > 0.2:  # the rewrite changed words, not just script: keep the original
                dropped["rewrite_drift"] += 1
                text = raw
            path = out / "wavs" / f"{cid}.wav"
            path.write_bytes(data)
            rows.append({"audio": str(path), "text": text, "duration": round(len(clip) / SR, 2),
                         "raw_text": raw, "speaker_sim": round(sim, 3), "rewrite_drift": round(drift, 3)})

    t0 = time.monotonic()
    jobs = []
    for si, src in enumerate(args.sources):
        x, sr = sf.read(src, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            x = soxr.resample(x, sr, SR)
        x16 = soxr.resample(x, SR, 16000).astype(np.float32)
        clips = pack(segments(x16), x16)
        print(f"{src}: {len(x)/SR/60:.1f} min -> {len(clips)} candidate clips, {sum(b-a for a,b in clips)/60:.1f} min")
        for ci, (a, b) in enumerate(clips):
            a0, b0 = max(0, int((a - 0.05) * SR)), min(len(x), int((b + 0.15) * SR))
            clip = x[a0:b0].copy()
            fade = int(0.01 * SR)
            clip[:fade] *= np.linspace(0, 1, fade)
            clip[-fade:] *= np.linspace(1, 0, fade)
            # Loudness: bring every clip to about -20 dBFS RMS, never above -1 dBFS peak.
            gain = 10 ** ((-20 - dbfs(clip)) / 20)
            gain = min(gain, 0.89 / (np.max(np.abs(clip)) + 1e-9))
            jobs.append(handle(f"s{si:02d}_{ci:04d}", (clip * gain).astype(np.float32)))
    await asyncio.gather(*jobs)
    rows.sort(key=lambda r: r["audio"])

    n_val = max(2, int(len(rows) * args.val_fraction))
    val, train = rows[::max(1, len(rows) // n_val)][:n_val], None
    val_set = {r["audio"] for r in val}
    train = [r for r in rows if r["audio"] not in val_set]
    for name, part in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({"audio": r["audio"], "text": r["text"], "duration": r["duration"]}, ensure_ascii=False) + "\n")
    with open(out / "all_with_details.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mins = sum(r["duration"] for r in rows) / 60
    report = {"clips": len(rows), "minutes": round(mins, 1), "train": len(train), "val": len(val), "dropped": dropped,
              "speaker_sim_median": float(np.median([r["speaker_sim"] for r in rows])) if rows else None,
              "seconds_taken": round(time.monotonic() - t0)}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
