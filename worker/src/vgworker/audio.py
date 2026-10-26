import io

import numpy as np
import soundfile as sf
import soxr


def decode(data: bytes, target_sr: int) -> np.ndarray:
    """Decode any soundfile-readable container (wav, flac, ogg) to mono float32 at target_sr."""
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    mono = wav.mean(axis=1)
    if sr != target_sr:
        mono = soxr.resample(mono, sr, target_sr)
    return mono.astype(np.float32)


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def write_wav(path, x: np.ndarray, sr: int) -> None:
    sf.write(path, x, sr, subtype="PCM_16")
