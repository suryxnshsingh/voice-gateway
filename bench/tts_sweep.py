"""TTS settings sweep on the live worker: speed vs intelligibility vs voice similarity.

    cd bench && uv run python tts_sweep.py --voice voice_19d9ec2eae586ac7 --real ../data-raw/screen_recording/ritesh_clean.wav

For each (inference_timesteps, cfg_value) it renders the same tutor sentences and reports first-chunk
latency, real-time factor, transcription error (gpt-transcribe, script-agnostic skeleton) and ECAPA
similarity to a real recording of the speaker. Samples land in --out for listening.
"""

import argparse
import asyncio
import io
import json
import statistics
import wave
from pathlib import Path

import httpx
import websockets
from openai import AsyncOpenAI

from stt_bench import score, settings

SENTENCES = [
    "देखो बच्चों, mitochondria को powerhouse of the cell इसलिए बोलते हैं क्योंकि यही cell के लिए energy बनाता है।",
    "अच्छा सवाल है! DNA replication semi-conservative होता है, मतलब हर नए DNA में एक strand पुराना और एक नया होता है।",
    "कोई बात नहीं बेटा, एक बार फिर से समझते हैं। Calvin cycle में carbon dioxide से glucose बनता है।",
]


async def render(voice, text, steps, cfg, sr=48000) -> tuple[bytes, dict]:
    pcm = bytearray()
    async with websockets.connect(settings.worker_url.replace("http", "ws") + "/tts", max_size=2**24) as ws:
        await ws.send(json.dumps({"voice_id": voice, "text": text, "sample_rate": sr,
                                  "inference_timesteps": steps, "cfg_value": cfg}))
        async for m in ws:
            if isinstance(m, bytes):
                pcm += m
            else:
                return bytes(pcm), json.loads(m)


def wav(pcm: bytes, sr=48000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(pcm)
    return buf.getvalue()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", required=True)
    ap.add_argument("--real", required=True, help="a real recording of the speaker, for similarity")
    ap.add_argument("--configs", default="10:2.0,8:2.0,6:2.0,4:2.0,8:1.5,8:2.5")
    ap.add_argument("--out", default=str(Path.home() / "Desktop" / "ritesh_voice_ab" / "sweep"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    real = Path(args.real).read_bytes()
    oai = AsyncOpenAI(api_key=settings.openai_key)
    http = httpx.AsyncClient(base_url=settings.worker_url, timeout=120)
    await render(args.voice, "नमस्ते।", 10, 2.0)  # warm
    print(f"{'steps':>5} {'cfg':>4} {'first ms':>9} {'RTF':>5} {'asr err':>8} {'voice sim':>9}")
    for cfgstr in args.configs.split(","):
        steps, cfg = int(cfgstr.split(":")[0]), float(cfgstr.split(":")[1])
        firsts, rtfs, errs, sims = [], [], [], []
        for i, text in enumerate(SENTENCES):
            pcm, stats = await render(args.voice, text, steps, cfg)
            data = wav(pcm)
            (out / f"steps{steps}_cfg{cfg}_{i}.wav").write_bytes(data)
            firsts.append(stats["first_chunk_ms"]); rtfs.append(stats["total_ms"] / stats["audio_ms"])
            heard = (await oai.audio.transcriptions.create(model="gpt-transcribe", file=("s.wav", data))).text
            errs.append(score(text, heard)["skel_error"])
            r = await http.post("/speaker/similarity", files={"a": ("a.wav", real), "b": ("b.wav", data)})
            sims.append(r.json()["score"])
        print(f"{steps:>5} {cfg:>4} {statistics.median(firsts):>9.0f} {statistics.median(rtfs):>5.2f} "
              f"{statistics.mean(errs):>8.3f} {statistics.mean(sims):>9.3f}")


if __name__ == "__main__":
    asyncio.run(main())
