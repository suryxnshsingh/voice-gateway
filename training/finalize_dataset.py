"""Second pass over a prepared dataset, from the fine-tuning research:

- Digits: the model can't know whether "51" was said "इक्यावन" or "fifty one". Re-transcribe those
  clips asking for numbers as spoken words; drop any that still have digits.
- A trailing "..." (clip cut mid-sentence) becomes ","; text is NFC-normalized.
- Paths become absolute (the trainer resolves relative paths against the manifest's folder).
- About half the training rows get a short reference clip ("ref_audio"), because inference clones
  from a reference (worker: build_prompt_cache(reference_wav_path=...)).

    cd bench && uv run python ../training/finalize_dataset.py ../training/datasets/ritesh
"""

import asyncio
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bench"))
from stt_bench import settings  # noqa: E402

PROMPT = ("Arivihan, Ritesh Sir, MP Board, Unnati Batch, NCERT. Write every number as words exactly as spoken, "
          "in the language it was spoken (इक्यावन, तीस हज़ार, twelfth), never as digits.")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text.strip())
    return re.sub(r"\s*(\.\.\.|…)\s*$", ",", text)


async def main(folder: Path):
    oai = AsyncOpenAI(api_key=settings.openai_key)
    rows = [json.loads(l) for l in open(folder / "all_with_details.jsonl", encoding="utf-8")]
    sem = asyncio.Semaphore(8)
    dropped = []

    async def fix_digits(r):
        async with sem:
            with open(r["audio"], "rb") as f:
                heard = (await oai.audio.transcriptions.create(model="gpt-transcribe", file=f, prompt=PROMPT)).text
            if re.search(r"\d", heard):
                dropped.append(r["audio"])
                return
            # Same script rule as the first pass: English words in Latin letters.
            resp = await oai.chat.completions.create(model="gpt-5.4-mini", messages=[
                {"role": "system", "content": "Rewrite this Hindi/Hinglish transcript so every English word is in English "
                 "letters and every Hindi word stays in Devanagari. Write the brand as Arivihan. Do not change, add or "
                 "remove words. Reply with the transcript only."},
                {"role": "user", "content": heard}])
            r["text"] = resp.choices[0].message.content.strip()
            r["renumbered"] = True

    await asyncio.gather(*(fix_digits(r) for r in rows if re.search(r"\d", r["text"])))
    rows = [r for r in rows if r["audio"] not in dropped]
    for r in rows:
        r["text"] = clean(r["text"])
        r["audio"] = str(Path(r["audio"]).resolve())

    val_audio = {str(Path(json.loads(l)["audio"]).resolve()) for l in open(folder / "val.jsonl", encoding="utf-8")}
    train = [r for r in rows if r["audio"] not in val_audio]
    val = [r for r in rows if r["audio"] in val_audio]
    random.seed(0)
    short = [r for r in train if r["duration"] <= 8]
    for r in train:
        if random.random() < 0.5:
            ref = random.choice([s for s in short if s is not r])
            r["ref_audio"] = ref["audio"]
    keep = ("audio", "text", "duration", "ref_audio")
    for name, part in (("train_final", train), ("val_final", val)):
        with open(folder / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps({k: r[k] for k in keep if k in r}, ensure_ascii=False) + "\n")
    print(json.dumps({"train": len(train), "val": len(val), "minutes": round(sum(r["duration"] for r in train) / 60, 1),
                      "renumbered": sum(1 for r in rows if r.get("renumbered")), "dropped_digits": len(dropped),
                      "with_ref_audio": sum(1 for r in train if "ref_audio" in r)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]).resolve()))
