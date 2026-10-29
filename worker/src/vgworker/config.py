import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"VG_WORKER_{name}", default)


@dataclass(frozen=True)
class Settings:
    # "auto" lets VoxCPM pick cuda > mps > cpu.
    device: str = field(default_factory=lambda: _env("DEVICE", "auto"))
    # One VoxCPM2 instance renders one sentence at a time (its KV cache has batch size 1),
    # so concurrent sentences need more instances. Roughly 5 GB of VRAM each in bf16.
    tts_instances: int = field(default_factory=lambda: int(_env("TTS_INSTANCES", "1")))
    tts_model: str = field(default_factory=lambda: _env("TTS_MODEL", "openbmb/VoxCPM2"))
    tts_cfg: float = field(default_factory=lambda: float(_env("TTS_CFG", "2.0")))
    # 8 diffusion steps: 12% faster than VoxCPM's default 10 with the same voice similarity and
    # intelligibility (bench/tts_sweep.py on an M5 Pro, Oct 2026). 4 audibly degrades.
    tts_timesteps: int = field(default_factory=lambda: int(_env("TTS_TIMESTEPS", "8")))
    tts_max_chars: int = field(default_factory=lambda: int(_env("TTS_MAX_CHARS", "600")))
    # Fine-tuned voices are LoRA adapters on the one base model, switched per sentence: about 70 MB
    # of RAM each, where a merged fine-tune is a whole model on the GPU. Every adapter must be
    # trained with this rank and alpha on the LM and DiT attention (training/conf: r 32, alpha 32).
    # LORA_R=0 builds the model without LoRA layers, for serving a merged checkpoint as before.
    lora_r: int = field(default_factory=lambda: int(_env("LORA_R", "32")))
    lora_alpha: int = field(default_factory=lambda: int(_env("LORA_ALPHA", "32")))
    # How many adapters stay in RAM; the rest load from disk (about 0.1 s) when next used.
    adapter_cache: int = field(default_factory=lambda: int(_env("ADAPTER_CACHE", "16")))
    # Seconds of idleness after which the worker renders a tiny word to keep the model in memory.
    # 0 = off (GPU servers). On a Mac under memory pressure, ~45 avoids 10-30 s first-request stalls.
    keep_warm_s: int = field(default_factory=lambda: int(_env("KEEP_WARM_S", "0")))
    tone_model: str = field(default_factory=lambda: _env("TONE_MODEL", "emotion2vec/emotion2vec_plus_base"))
    tone_device: str = field(default_factory=lambda: _env("TONE_DEVICE", "cpu"))
    speaker_model: str = field(default_factory=lambda: _env("SPEAKER_MODEL", "speechbrain/spkrec-ecapa-voxceleb"))
    speaker_device: str = field(default_factory=lambda: _env("SPEAKER_DEVICE", "cpu"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./data")).resolve())
    # Shared secret the gateway sends as x-worker-token. Empty disables the check (local dev only).
    token: str = field(default_factory=lambda: _env("TOKEN", ""))
    # Comma-separated subset of tts,tone,speaker to load. Lets you run e.g. tone-only on a CPU box.
    load: frozenset[str] = field(
        default_factory=lambda: frozenset(s.strip() for s in _env("LOAD", "tts,tone,speaker").split(",") if s.strip())
    )
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8100")))

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def adapters_dir(self) -> Path:
        """One folder per adapter, holding the lora_weights.safetensors training writes."""
        return self.data_dir / "adapters"


settings = Settings()
