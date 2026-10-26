"""Score eval_checkpoints.py renders: voice similarity to real held-out clips (ECAPA, via the worker)
and intelligibility (gpt-transcribe back to text, script-agnostic skeleton error).

    cd bench && uv run python ../training/score_eval.py ../training/eval/ritesh_r32 --real ../training/datasets/ritesh/val_final.jsonl
"""

import argparse
import asyncio
import collections
import json
import statistics
import sys
from pathlib import Path

import httpx
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from stt_bench import score, settings  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_dir")
    ap.add_argument("--real", required=True, help="manifest of real clips NOT used in training (val)")
    args = ap.parse_args()
    d = Path(args.eval_dir)
    sentences = json.loads((d / "sentences.json").read_text())
    timing = json.loads((d / "timing.json").read_text())
    real = [json.loads(l)["audio"] for l in open(args.real, encoding="utf-8")]
    real_bytes = [Path(p).read_bytes() for p in real[:3]]
    http = httpx.AsyncClient(base_url=settings.worker_url, timeout=120)
    oai = AsyncOpenAI(api_key=settings.openai_key)
    sem = asyncio.Semaphore(6)
    results = collections.defaultdict(lambda: {"sim": [], "err": [], "rtf": []})

    async def one(wav: Path):
        variant, ref, i = wav.stem.split("__")
        data = wav.read_bytes()
        async with sem:
            sims = []
            for rb in real_bytes:
                r = await http.post("/speaker/similarity", files={"a": ("a.wav", rb), "b": ("b.wav", data)})
                sims.append(r.json()["score"])
            heard = (await oai.audio.transcriptions.create(model="gpt-transcribe", file=("x.wav", data))).text
        key = (variant, ref)
        results[key]["sim"].append(statistics.mean(sims))
        results[key]["err"].append(score(sentences[int(i)], heard)["skel_error"])
        t = timing[wav.stem]
        results[key]["rtf"].append(t["seconds"] / max(t["audio_s"], 0.1))

    await asyncio.gather(*(one(w) for w in sorted(d.glob("*__*__*.wav"))))
    print(f"{'variant':16s} {'ref':6s} {'voice sim':>9s} {'asr err':>8s} {'RTF':>5s}")
    rows = []
    for (variant, ref), r in sorted(results.items()):
        row = {"variant": variant, "ref": ref, "sim": round(statistics.mean(r["sim"]), 3),
               "err": round(statistics.mean(r["err"]), 3), "rtf": round(statistics.mean(r["rtf"]), 2)}
        rows.append(row)
        print(f"{variant:16s} {ref:6s} {row['sim']:>9.3f} {row['err']:>8.3f} {row['rtf']:>5.2f}")
    (d / "scores.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
