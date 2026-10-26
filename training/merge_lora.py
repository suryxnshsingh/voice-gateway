"""Merge a VoxCPM2 LoRA checkpoint into the base weights -> a plain VoxCPM2 model dir.
usage: python merge_lora.py BASE_DIR LORA_CKPT_DIR OUT_DIR"""
import json, shutil, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

base, ckpt, out = map(Path, sys.argv[1:4])
cfg = json.loads((ckpt / "lora_config.json").read_text())["lora_config"]
scale = cfg["alpha"] / cfg["r"]
W = load_file(str(base / "model.safetensors"))          # bf16 tensors, ~4.6 GB in RAM
L = load_file(str(ckpt / "lora_weights.safetensors"))
n = 0
for k in [k for k in L if k.endswith(".lora_A")]:
    p = k[: -len(".lora_A")]
    A, B = L[k].float(), L[p + ".lora_B"].float()
    W[p + ".weight"] = (W[p + ".weight"].float() + scale * (B @ A)).contiguous()  # keep merged tensors fp32
    n += 1
assert n == len(L) // 2, (n, len(L))
out.mkdir(parents=True, exist_ok=True)
save_file(W, str(out / "model.safetensors"), metadata={"format": "pt"})
for f in base.iterdir():
    if f.name not in ("model.safetensors",) and f.is_file():
        shutil.copy2(f, out / f.name)
print(f"merged {n} LoRA modules (scale={scale}) -> {out}")
