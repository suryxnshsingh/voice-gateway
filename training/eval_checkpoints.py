"""Render held-out sentences from the base model and each LoRA checkpoint, for scoring.

    cd training && voxcpm-src/.venv/bin/python eval_checkpoints.py ckpt/ritesh_r32 --ref-long R30.wav --ref-short R8.wav

Writes eval/<run>/<variant>__<ref>__<n>.wav plus timing.json. Score them with score_eval.py (needs the
worker for ECAPA similarity). Sentences are tutoring lines Ritesh Sir never said in the training audio.
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import soundfile as sf  # noqa: E402
from voxcpm import VoxCPM  # noqa: E402
from voxcpm.model.voxcpm2 import LoRAConfig  # noqa: E402

SENTENCES = [
    "देखो बच्चों, mitochondria को powerhouse of the cell इसलिए बोलते हैं, क्योंकि यही cell के लिए energy बनाता है।",
    "अच्छा सवाल है बेटा! DNA replication semi-conservative होता है, मतलब हर नए DNA में एक strand पुराना होता है।",
    "कोई बात नहीं, एक बार फिर से समझते हैं। Calvin cycle में carbon dioxide से glucose बनता है, ठीक है?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--ref-long", required=True)
    ap.add_argument("--ref-short", required=True)
    ap.add_argument("--steps", default="", help="comma-separated steps to evaluate (default: all saved)")
    ap.add_argument("--out", default="eval")
    args = ap.parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(p for p in ckpt_dir.glob("step_*") if (p / "lora_weights.safetensors").exists())
    if args.steps:
        wanted = {int(s) for s in args.steps.split(",")}
        ckpts = [p for p in ckpts if int(p.name.split("_")[1]) in wanted]
    cfg = json.loads((ckpts[0] / "lora_config.json").read_text())
    model = VoxCPM.from_pretrained(cfg["base_model"], load_denoiser=False, optimize=False,
                                   lora_config=LoRAConfig(**cfg["lora_config"]), lora_weights_path=str(ckpts[0]))
    out = Path(args.out) / ckpt_dir.name
    out.mkdir(parents=True, exist_ok=True)
    variants = [("base", None)] + [(p.name, p) for p in ckpts if p.name != "step_0000000"]
    timing = {}
    for name, path in variants:
        if path is None:
            model.set_lora_enabled(False)
        else:
            model.load_lora(str(path))
            model.set_lora_enabled(True)
        for ref_name, ref in (("long", args.ref_long), ("short", args.ref_short)):
            for i, text in enumerate(SENTENCES):
                t0 = time.monotonic()
                wav = model.generate(text=text, reference_wav_path=ref, inference_timesteps=10, cfg_value=2.0)
                dt = time.monotonic() - t0
                sf.write(out / f"{name}__{ref_name}__{i}.wav", wav, model.tts_model.sample_rate)
                timing[f"{name}__{ref_name}__{i}"] = {"seconds": round(dt, 2), "audio_s": round(len(wav) / model.tts_model.sample_rate, 2)}
                print(f"{name} {ref_name} {i}: {dt:.1f}s for {len(wav) / model.tts_model.sample_rate:.1f}s audio", flush=True)
    (out / "timing.json").write_text(json.dumps(timing, indent=2))
    (out / "sentences.json").write_text(json.dumps(SENTENCES, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
