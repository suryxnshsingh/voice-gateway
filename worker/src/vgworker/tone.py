"""Acoustic tone: emotion2vec+ utterance-level emotion scores from 16 kHz audio."""

import threading

import numpy as np

# emotion2vec labels come as "中文/english"; <unk> carries no signal.
_SKIP = {"<unk>"}


class ToneModel:
    def __init__(self, model_id: str, device: str):
        from funasr import AutoModel

        self._model = AutoModel(model=model_id, hub="hf", device=device, disable_update=True, disable_pbar=True)
        self._lock = threading.Lock()

    def classify(self, audio16k: np.ndarray) -> dict:
        with self._lock:
            res = self._model.generate(audio16k, granularity="utterance", extract_embedding=False, disable_pbar=True)[0]
        scores = {}
        for label, score in zip(res["labels"], res["scores"]):
            if label in _SKIP:
                continue
            scores[label.split("/")[-1]] = round(float(score), 4)
        label = max(scores, key=scores.get)
        return {"label": label, "confidence": scores[label], "scores": scores}
