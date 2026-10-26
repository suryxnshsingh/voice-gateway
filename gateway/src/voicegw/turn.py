"""Semantic end-of-turn: Pipecat's Smart Turn v3.2 (BSD-2, 8 MB ONNX, Whisper-tiny encoder).

It listens to the last 8 s of the user's audio and says whether they sound finished, from intonation
and rhythm rather than silence length. Hindi accuracy 90.1% (FNR 1.3%) in Pipecat's v3.2 benchmark.
Used with Silero: at a VAD stop, a confident "finished" ends the turn at once; otherwise the turn waits
for the longer silence fallback, so a thinking pause ("मतलब...") isn't cut off.

The features are Whisper's log-mel (what transformers' WhisperFeatureExtractor(chunk_length=8) makes),
reimplemented in numpy so the gateway doesn't need transformers; tests/test_turn.py checks parity.
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

SR = 16000
N_FFT = 400
HOP = 160
N_MELS = 80
WINDOW_S = 8
MODEL_PATH = Path(__file__).parent / "models" / "smart-turn-v3.2-cpu.onnx"


def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, f / f_sp)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel, logstep = min_log_hz / f_sp, np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)


def _mel_filters() -> np.ndarray:
    """(n_freq, n_mels) Slaney-normalised triangular filters, as in transformers.audio_utils.mel_filter_bank."""
    fft_freqs = np.linspace(0, SR // 2, 1 + N_FFT // 2)
    mel_freqs = np.linspace(_hz_to_mel(0.0), _hz_to_mel(8000.0), N_MELS + 2)
    filter_freqs = _mel_to_hz(mel_freqs)
    diff = np.diff(filter_freqs)
    slopes = filter_freqs[None, :] - fft_freqs[:, None]
    down = -slopes[:, :-2] / diff[:-1]
    up = slopes[:, 2:] / diff[1:]
    filters = np.maximum(0, np.minimum(down, up))
    return filters * (2.0 / (filter_freqs[2 : N_MELS + 2] - filter_freqs[:N_MELS]))[None, :]


_FILTERS = _mel_filters()
_WINDOW = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)  # periodic Hann


def features(audio16k: np.ndarray) -> np.ndarray:
    """Last 8 s (left-padded with zeros) -> normalised -> (1, 80, 800) Whisper log-mel."""
    n = WINDOW_S * SR
    x = audio16k[-n:].astype(np.float64)
    x = np.pad(x, (n - len(x), 0))
    x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    x = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    frames = np.lib.stride_tricks.sliding_window_view(x, N_FFT)[::HOP] * _WINDOW
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2  # (frames, 201)
    mel = np.maximum(power @ _FILTERS, 1e-10).T  # (80, frames)
    log_spec = np.log10(mel)[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)[None]


class SmartTurn:
    _session: ort.InferenceSession | None = None

    def __init__(self):
        if SmartTurn._session is None:
            opts = ort.SessionOptions()
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            SmartTurn._session = ort.InferenceSession(str(MODEL_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
        self._sess = SmartTurn._session

    def p_complete(self, audio16k: np.ndarray) -> float:
        return float(self._sess.run(None, {"input_features": features(audio16k)})[0].reshape(-1)[0])
