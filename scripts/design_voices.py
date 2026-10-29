"""Design the shared voices: new voices described in words with VoxCPM2 voice design, so no real
person's voice is used, then checked by cloning them through the production worker.

Steps (DIR holds candidates, scores and the chosen ids between steps):

    # on the GPU VM, with the worker's venv; a temporary second model, base weights only (~7 GB)
    worker/.venv/bin/python scripts/design_voices.py generate DIR --seeds 12
    # on the VM: ECAPA similarity to AVOID (the fine-tuned speaker) and the longest pause (stalls)
    worker/.venv/bin/python scripts/design_voices.py score DIR
    # on a Mac, with bench/.venv (mlx_whisper): does the take say the text
    bench/.venv/bin/python scripts/design_voices.py asr DIR
    # on the VM: clone shortlisted takes through the production worker, scored on unseen lines
    worker/.venv/bin/python scripts/design_voices.py clone DIR riya:1003 ananya:1001 ...
    # on the VM: give each pick an id, master its reference into the gateway's data and render
    # its preview through the production worker
    worker/.venv/bin/python scripts/design_voices.py finalize DIR riya:1003 ananya:1001 ...

Scoring steps talk to the worker at WORKER (127.0.0.1:8100) with VG_WORKER_TOKEN.
"""

import argparse
import asyncio
import json
import os
import re
import secrets
import subprocess
import sys
import time
import wave
from pathlib import Path

# Each voice: the design prompt, a ~15 s reference script in its own register, and the ~5 s preview
# customers hear. Generic synthetic voices, chosen to sound clearly different from each other.
VOICES = {
    "riya": {
        "name": "Riya", "gender": "female",
        "description": "Bright, upbeat young woman; clear and friendly, for app updates and notifications.",
        "design": "A bright, friendly young Indian woman, clear and upbeat, speaking at a natural pace",
        "text": "नमस्ते! आपका order आज शाम छह बजे तक deliver हो जाएगा। Rider का number SMS में भेज दिया गया है। "
                "अगर आप घर पर नहीं हैं, तो app में जाकर delivery का time बदल सकती हैं। Thank you, आपका दिन शुभ हो!",
        "preview": "नमस्ते, मैं Riya हूँ। आपका order आज शाम तक आपके घर पहुँच जाएगा।",
    },
    "ananya": {
        "name": "Ananya", "gender": "female",
        "description": "Warm, patient woman in her thirties; gentle and easy to follow, for teaching and explainers.",
        "design": "A warm Indian woman in her thirties, a patient school teacher explaining to a student",
        "text": "देखो बेटा, mitochondria को cell का powerhouse कहते हैं, क्योंकि energy यहीं बनती है। "
                "अब एक आसान example लेते हैं। जैसे हमारे घर में बिजली meter से आती है, वैसे ही cell को energy "
                "mitochondria से मिलती है। समझ आया? चलो, अगला topic शुरू करते हैं।",
        "preview": "नमस्ते, मैं Ananya हूँ। चलिए, आज का chapter आसान examples से समझते हैं।",
    },
    "meera": {
        "name": "Meera", "gender": "female",
        "description": "Crisp, composed woman in her forties; measured and confident, for news, narration and IVR.",
        "design": "A mature, composed Indian woman in her early forties, a news presenter with a smooth, "
                  "slightly low voice, measured and confident, speaking at a steady pace",
        "text": "नमस्कार, आज की मुख्य खबरों में सबसे पहले मौसम का हाल। Mumbai और Pune में अगले दो दिनों तक तेज़ बारिश "
                "की संभावना है, इसलिए घर से निकलने से पहले traffic update ज़रूर देख लें। "
                "बाकी खबरें हम आपको एक छोटे से break के बाद बताएँगे।",
        "preview": "नमस्कार, मैं Meera हूँ। अब सुनिए आज की ताज़ा खबरें, विस्तार से।",
    },
    "kabir": {
        "name": "Kabir", "gender": "male",
        "description": "Energetic young man in his early twenties; lively and expressive, for promos and social videos.",
        "design": "An energetic, cheerful young Indian man in his early twenties, a lively YouTube host, "
                  "crisp and expressive, speaking at a brisk pace",
        "text": "Hey guys, welcome back! आज हम बात करेंगे पाँच ऐसे apps की, जो आपकी daily life को super easy बना देंगे। "
                "तो video को end तक ज़रूर देखना, और अगर पसंद आए, तो like और subscribe करना मत भूलना। "
                "चलो, शुरू करते हैं!",
        "preview": "हाय, मैं Kabir हूँ! चलो, आज कुछ नया और मज़ेदार try करते हैं।",
    },
    "dev": {
        "name": "Dev", "gender": "male",
        "description": "Calm, reassuring man in his thirties; professional and clear, for support and banking.",
        "design": "A composed, reassuring Indian man in his thirties, professional and clear",
        "text": "नमस्ते, आपकी EMI पाँच तारीख को due है, और amount है बारह हज़ार चार सौ रुपये। "
                "क्या मैं payment link WhatsApp पर भेज दूँ? Payment होते ही आपको confirmation message मिल जाएगा। "
                "किसी भी सवाल के लिए आप हमें कभी भी call कर सकते हैं।",
        "preview": "नमस्ते, मैं देव हूँ। आपकी EMI की details मैंने WhatsApp पर भेज दी हैं।",
    },
    "vikram": {
        "name": "Vikram", "gender": "male",
        "description": "Deep, confident man in his forties; a rich announcer voice for ads, announcements and narration.",
        "design": "A deep-voiced Indian man in his forties, a confident radio announcer, rich and warm, "
                  "speaking clearly at a steady pace",
        "text": "दोस्तों, इस festive season में लाइए अपने घर नई खुशियाँ। हर smartphone पर पाइए flat पच्चीस percent "
                "की छूट, और साथ में free delivery भी। यह offer सिर्फ़ इस रविवार तक है, तो देर मत कीजिए। "
                "आज ही visit कीजिए हमारा store, या order कीजिए online।",
        "preview": "नमस्कार, मैं Vikram हूँ। इस festive season, बड़ी बचत का मौका मत छोड़िए।",
    },
}

# Lines no reference contains: a clone is judged on text it never heard.
CLONE_LINES = [
    "नमस्ते, आपका order कल सुबह तक पहुँच जाएगा। कोई सवाल हो तो हमें बताइए।",
    "Meeting का time बदलकर चार बजे कर दिया गया है, please सबको inform कर दीजिए।",
]
AVOID = Path.home() / "voice-gateway/gateway/data/voices/voice_19d9ec2eae586ac7/reference.wav"
GATEWAY_VOICES = Path.home() / "voice-gateway/gateway/data/voices"
WORKER = os.environ.get("WORKER", "127.0.0.1:8100")
SR = 48000


def _report(d: Path) -> dict:
    p = d / "candidates.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save(d: Path, report: dict) -> None:
    (d / "candidates.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


def _picks(args) -> list[tuple[str, int]]:
    out = []
    for p in args.picks:
        voice, seed = p.split(":")
        if voice not in VOICES:
            raise SystemExit(f"unknown voice {voice}")
        out.append((voice, int(seed)))
    return out


def measure(wav, sr: int) -> dict:
    """Length, median voiced pitch (does it match the name's gender) and level."""
    import librosa
    import numpy as np

    y = librosa.resample(wav, orig_sr=sr, target_sr=16000) if sr != 16000 else wav
    f0, voiced, _ = librosa.pyin(y, fmin=65, fmax=400, sr=16000, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    return {
        "seconds": round(len(wav) / sr, 2),
        "f0_median": round(float(np.median(f0)), 1) if len(f0) else None,
        "peak": round(float(np.max(np.abs(wav))), 3),
        "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(wav**2)) + 1e-9)), 1),
    }


def master(src: Path, dst: Path, mp3: bool = False) -> None:
    """Trim silence at both ends, keep 120 ms of air, normalise to -16 LUFS, 48 kHz mono
    (the same chain as the site's samples)."""
    trim = "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.12"
    codec = ["-codec:a", "libmp3lame", "-b:a", "128k"] if mp3 else ["-codec:a", "pcm_s16le"]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                    "-af", f"{trim},areverse,{trim},areverse,loudnorm=I=-16:TP=-1.5:LRA=11",
                    "-ar", str(SR), "-ac", "1", *codec, str(dst)], check=True)


# --- worker calls (production worker; only base weights, since no adapter is named) ---

def _headers() -> dict:
    return {"x-worker-token": os.environ["VG_WORKER_TOKEN"]}


def similarity(a: Path, b: Path) -> float:
    import httpx

    with open(a, "rb") as fa, open(b, "rb") as fb:
        r = httpx.post(f"http://{WORKER}/speaker/similarity", headers=_headers(), files={"a": fa, "b": fb}, timeout=60)
    r.raise_for_status()
    return r.json()["score"]


def put_voice(voice_id: str, audio: Path) -> dict:
    import httpx

    r = httpx.put(f"http://{WORKER}/voices/{voice_id}", headers=_headers(), content=audio.read_bytes(), timeout=120)
    r.raise_for_status()
    return r.json()


def delete_voice(voice_id: str) -> None:
    import httpx

    httpx.delete(f"http://{WORKER}/voices/{voice_id}", headers=_headers(), timeout=30)


def render(voice_id: str, text: str, out: Path) -> dict:
    """One sentence through WS /tts, saved as a 48 kHz wav. Returns the worker's stats."""
    import websockets

    async def run():
        pcm = bytearray()
        async with websockets.connect(f"ws://{WORKER}/tts", additional_headers=_headers(), max_size=None) as ws:
            await ws.send(json.dumps({"voice_id": voice_id, "text": text, "sample_rate": SR}))
            async for msg in ws:
                if isinstance(msg, bytes):
                    pcm.extend(msg)
                    continue
                info = json.loads(msg)
                if info.get("type") != "done":
                    raise RuntimeError(f"worker: {info}")
                with wave.open(str(out), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes(bytes(pcm))
                return info

    return asyncio.run(run())


# --- steps ---

def cmd_generate(args) -> None:
    import numpy as np
    import soundfile as sf
    import torch
    from voxcpm import VoxCPM

    args.dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False, optimize=False, device=args.device)
    sr = model.tts_model.sample_rate
    print(f"base model ready in {time.monotonic() - t0:.0f} s", file=sys.stderr)
    report = _report(args.dir)
    for voice in args.only.split(",") if args.only else VOICES:
        v = VOICES[voice]
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            started = time.monotonic()
            wav = model.generate(text=f"({v['design']}){v['text']}", cfg_value=args.cfg, inference_timesteps=args.timesteps)
            path = args.dir / f"{voice}-{seed}.wav"
            sf.write(path, wav, sr)
            c = {"seed": seed, "path": path.name, "render_s": round(time.monotonic() - started, 1)} | measure(wav, sr)
            c["chars_per_s"] = round(len(v["text"]) / c["seconds"], 1)
            report[voice] = [x for x in report.get(voice, []) if x["seed"] != seed] + [c]
            _save(args.dir, report)
            print(f"{voice}-{seed}: {json.dumps(c)}", file=sys.stderr)


def longest_pause(path: Path) -> float:
    """Seconds of the longest quiet stretch inside the speech (40 dB under its loudest 20 ms):
    a design take that stalls or trails off mid sentence makes a poor reference."""
    import numpy as np
    import soundfile as sf

    wav, sr = sf.read(path)
    hop = sr // 50
    frames = wav[: len(wav) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    loud = np.flatnonzero(db > db.max() - 40)
    quiet, run = 0, 0
    for is_loud in db[loud[0]:loud[-1] + 1] > db.max() - 40:
        run = 0 if is_loud else run + 1
        quiet = max(quiet, run)
    return round(quiet * hop / sr, 2)


def cmd_score(args) -> None:
    report = _report(args.dir)
    for voice, cands in report.items():
        for c in cands:
            path = args.dir / c["path"]
            c["vs_avoid"] = similarity(path, AVOID)
            c["max_pause_s"] = longest_pause(path)
            print(f"{voice}-{c['seed']}: vs_avoid {c['vs_avoid']}, pause {c['max_pause_s']} s, "
                  f"f0 {c['f0_median']}, {c['seconds']} s, peak {c['peak']}", file=sys.stderr)
        _save(args.dir, report)


def cmd_asr(args) -> None:
    import mlx_whisper
    from indic_transliteration import sanscript

    def skeleton(text: str) -> str:  # same as bench/stt_bench.py: script-agnostic consonants
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

    report = _report(args.dir)
    for voice, cands in report.items():
        ref = skeleton(VOICES[voice]["text"])
        for c in cands:
            heard = mlx_whisper.transcribe(str(args.dir / c["path"]), path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                                           language="hi", verbose=None)["text"].strip()
            c["heard"], c["skel_error"] = heard, round(edits(ref, skeleton(heard)) / max(len(ref), 1), 3)
            print(f"{voice}-{c['seed']}: err {c['skel_error']}: {heard}", file=sys.stderr)
        _save(args.dir, report)


def cmd_clone(args) -> None:
    """Master the take as a reference, clone it on the production worker under a temporary id,
    render unseen lines, and score them against the reference and against AVOID."""
    report = _report(args.dir)
    for voice, seed in _picks(args):
        c = next(x for x in report[voice] if x["seed"] == seed)
        ref = args.dir / f"{voice}-{seed}.ref.wav"
        master(args.dir / c["path"], ref)
        tmp = f"voice_tmpdesign_{voice}_{seed}"
        try:
            c["ref_seconds"] = put_voice(tmp, ref)["seconds"]
            selfs, avoids, firsts = [], [], []
            for i, line in enumerate(CLONE_LINES):
                out = args.dir / f"{voice}-{seed}.clone{i}.wav"
                firsts.append(render(tmp, line, out)["first_chunk_ms"])
                selfs.append(similarity(out, ref))
                avoids.append(similarity(out, AVOID))
        finally:
            delete_voice(tmp)
        c["clone_vs_ref"], c["clone_vs_avoid"] = round(sum(selfs) / len(selfs), 4), round(max(avoids), 4)
        _save(args.dir, report)
        print(f"{voice}-{seed}: clone~ref {selfs} clone~avoid {avoids} first_chunk_ms {firsts}", file=sys.stderr)


def cmd_finalize(args) -> None:
    """Each pick gets a voice id (kept across reruns), its reference goes where the gateway pushes
    it to workers from, and its preview is rendered by the production worker: what customers get."""
    report, ids_path = _report(args.dir), args.dir / "ids.json"
    ids = json.loads(ids_path.read_text()) if ids_path.exists() else {}
    previews = args.dir / "previews"
    previews.mkdir(exist_ok=True)
    manifest = []
    for voice, seed in _picks(args):
        v, c = VOICES[voice], next(x for x in report[voice] if x["seed"] == seed)
        voice_id = ids.setdefault(voice, "voice_" + secrets.token_hex(8))
        ids_path.write_text(json.dumps(ids, indent=2))
        ref = GATEWAY_VOICES / voice_id / "reference.wav"
        ref.parent.mkdir(parents=True, exist_ok=True)
        master(args.dir / c["path"], ref)
        seconds = put_voice(voice_id, ref)["seconds"]
        raw = previews / f"{voice_id}.raw.wav"
        stats = render(voice_id, v["preview"], raw)
        master(raw, previews / f"{voice_id}.mp3", mp3=True)
        manifest.append({"id": voice_id, "name": v["name"], "gender": v["gender"], "description": v["description"],
                         "reference_seconds": seconds, "seed": seed, "design": v["design"],
                         "preview_vs_ref": similarity(raw, ref), "preview_vs_avoid": similarity(raw, AVOID),
                         "preview_text": v["preview"], "preview_first_chunk_ms": stats["first_chunk_ms"]})
        print(json.dumps(manifest[-1], ensure_ascii=False), file=sys.stderr)
    # Shared voices should be told apart: every pair's references, scored against each other.
    refs = {m["name"]: GATEWAY_VOICES / m["id"] / "reference.wav" for m in manifest}
    names = list(refs)
    pairs = {f"{a}~{b}": similarity(refs[a], refs[b]) for i, a in enumerate(names) for b in names[i + 1:]}
    (args.dir / "manifest.json").write_text(json.dumps({"voices": manifest, "pairwise": pairs}, ensure_ascii=False, indent=2))
    print(json.dumps(pairs, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("dir", type=Path)
    g.add_argument("--seeds", type=int, default=8)
    g.add_argument("--seed-start", type=int, default=1000)
    g.add_argument("--only", default="", help="comma separated voice keys")
    g.add_argument("--timesteps", type=int, default=10)
    g.add_argument("--cfg", type=float, default=2.0)
    g.add_argument("--device", default="cuda")
    s = sub.add_parser("score")
    s.add_argument("dir", type=Path)
    a = sub.add_parser("asr")
    a.add_argument("dir", type=Path)
    for name in ("clone", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("dir", type=Path)
        p.add_argument("picks", nargs="+", help="voice:seed")
    args = ap.parse_args()
    {"generate": cmd_generate, "score": cmd_score, "asr": cmd_asr, "clone": cmd_clone, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    main()
