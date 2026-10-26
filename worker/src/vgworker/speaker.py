"""Speaker similarity (ECAPA-TDNN embeddings, cosine). Used by the consent check."""

import os
import threading

import numpy as np
import torch

# SpeechBrain's default quirks switch off TorchScript's profiling executor process-wide. The legacy
# executor then tries to fuse VoxCPM's scripted ops on MPS and fails with "Unknown device for graph fuser".
os.environ.setdefault("SB_DISABLE_QUIRKS", "disable_jit_profiling")

# Long clips add little to an embedding; cap the work.
_MAX_SECONDS = 30


class SpeakerModel:
    def __init__(self, model_id: str, device: str, savedir: str):
        from speechbrain.inference.speaker import EncoderClassifier

        self._enc = EncoderClassifier.from_hparams(source=model_id, savedir=savedir, run_opts={"device": device})
        self._lock = threading.Lock()

    def embed(self, audio16k: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(audio16k[: 16000 * _MAX_SECONDS]).unsqueeze(0)
        with self._lock, torch.inference_mode():
            return self._enc.encode_batch(x).squeeze().cpu()

    def similarity(self, a16k: np.ndarray, b16k: np.ndarray) -> float:
        return float(torch.nn.functional.cosine_similarity(self.embed(a16k), self.embed(b16k), dim=0))
