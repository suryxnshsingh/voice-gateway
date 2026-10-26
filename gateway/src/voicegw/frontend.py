"""Input audio front end: neural noise suppression (RNNoise) -> Silero VAD -> speech start/stop events.

Runs per session on the gateway's CPU. Both models are tiny: RNNoise is a ~100k-parameter GRU on
10 ms frames at 48 kHz, Silero VAD a ~2 MB ONNX graph on 32 ms windows at 16 kHz. A session costs a
few percent of one core.
"""

import ctypes
import importlib.util
import math
import platform
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soxr

SR = 16000
VAD_WINDOW = 512  # 32 ms; the window size Silero's 16 kHz graph is trained on
VAD_CONTEXT = 64
WINDOW_MS = VAD_WINDOW * 1000 // SR
SILERO_PATH = Path(__file__).parent / "models" / "silero_vad.onnx"


# --- RNNoise -------------------------------------------------------------------------------------

def _load_rnnoise() -> ctypes.CDLL:
    # Load the library pyrnnoise ships, without importing pyrnnoise itself (its package init pulls in
    # audio-file and plotting dependencies a server doesn't need).
    spec = importlib.util.find_spec("pyrnnoise")
    if spec is None or spec.origin is None:
        raise ImportError("pyrnnoise is not installed")
    name = {"Darwin": "librnnoise.dylib", "Linux": "librnnoise.so", "Windows": "rnnoise.dll"}[platform.system()]
    lib = ctypes.CDLL(str(Path(spec.origin).parent / name))
    lib.rnnoise_create.argtypes = [ctypes.c_void_p]
    lib.rnnoise_create.restype = ctypes.c_void_p
    lib.rnnoise_destroy.argtypes = [ctypes.c_void_p]
    lib.rnnoise_process_frame.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    lib.rnnoise_process_frame.restype = ctypes.c_float
    lib.rnnoise_get_frame_size.restype = ctypes.c_int
    return lib


_rnnoise: ctypes.CDLL | None = None


class NoiseSuppressor:
    """RNNoise works at 48 kHz; the stream is 16 kHz, so it runs between an up- and a down-sampler."""

    def __init__(self):
        global _rnnoise
        if _rnnoise is None:
            _rnnoise = _load_rnnoise()
        self._lib = _rnnoise
        self._state = self._lib.rnnoise_create(None)
        self._frame = self._lib.rnnoise_get_frame_size()  # 480 samples = 10 ms
        self._up = soxr.ResampleStream(SR, 48000, 1, dtype="float32")
        self._down = soxr.ResampleStream(48000, SR, 1, dtype="float32")
        self._pending = np.zeros(0, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        """float32 16 kHz in, denoised float32 16 kHz out (same length over time, ~10 ms behind)."""
        self._pending = np.concatenate([self._pending, self._up.resample_chunk(x)])
        n = len(self._pending) // self._frame * self._frame
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        # RNNoise expects int16-scaled floats and processes in place.
        block = np.ascontiguousarray(self._pending[:n] * 32768.0, dtype=np.float32)
        self._pending = self._pending[n:]
        for i in range(0, n, self._frame):
            ptr = block[i : i + self._frame].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self._lib.rnnoise_process_frame(self._state, ptr, ptr)
        return self._down.resample_chunk(block / 32768.0)

    def close(self) -> None:
        if self._state:
            self._lib.rnnoise_destroy(self._state)
            self._state = None


# --- Silero VAD ----------------------------------------------------------------------------------

_silero: ort.InferenceSession | None = None


def _silero_session() -> ort.InferenceSession:
    global _silero
    if _silero is None:
        opts = ort.SessionOptions()
        # Thousands of tiny calls: one thread each beats intra-op parallelism.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        _silero = ort.InferenceSession(str(SILERO_PATH), sess_options=opts, providers=["CPUExecutionProvider"])
    return _silero


class SileroVAD:
    """Streaming speech probability per 32 ms window. State is per instance; the ONNX session is shared."""

    def __init__(self):
        self._sess = _silero_session()
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(VAD_CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._sr = np.array(SR, dtype=np.int64)

    def process(self, x: np.ndarray) -> list[tuple[float, float]]:
        """Returns (speech probability, window RMS) for each complete window."""
        self._pending = np.concatenate([self._pending, x])
        out = []
        while len(self._pending) >= VAD_WINDOW:
            win, self._pending = self._pending[:VAD_WINDOW], self._pending[VAD_WINDOW:]
            inp = np.concatenate([self._context, win])[None, :]
            prob, self._state = self._sess.run(None, {"input": inp, "state": self._state, "sr": self._sr})
            self._context = win[-VAD_CONTEXT:]
            out.append((float(prob[0, 0]), float(np.sqrt(np.mean(win * win)) + 1e-9)))
        return out


# --- speech start/stop with hysteresis -----------------------------------------------------------

@dataclass
class VadParams:
    threshold: float = 0.6        # probability that counts as speech
    stop_threshold: float = 0.35  # below this counts as silence (hysteresis band in between)
    start_ms: int = 200           # speech must persist this long to start
    stop_ms: int = 500            # silence must persist this long to stop
    min_dbfs: float = -50.0       # ignore speech quieter than this (distant TV, people across the room)
    pause_ms: int = 200           # silence that raises a "pause" event (turn-end check) before the stop


@dataclass
class VadEvent:
    kind: str           # "start" | "pause" (short silence, still speaking) | "resume" (spoke again after a pause) | "stop"
    speech_ms: int = 0  # for "stop": how long the speech lasted


@dataclass
class FrontendResult:
    denoised: np.ndarray
    events: list[VadEvent] = field(default_factory=list)
    speaking: bool = False
    speech_ms: int = 0  # length of the current speech run, while speaking


class AudioFrontend:
    def __init__(self, params: VadParams, noise_suppression: bool = True):
        self.params = params
        self._ns = NoiseSuppressor() if noise_suppression else None
        self._vad = SileroVAD()
        self.speaking = False
        self._run_ms = 0      # consecutive ms on the far side of the current state
        self._speech_ms = 0   # length of the current speech segment
        self._paused = False  # a "pause" was raised in the current silence run
        self._min_rms = 10 ** (params.min_dbfs / 20)

    def process(self, pcm16: bytes) -> FrontendResult:
        x = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        clean = self._ns.process(x) if self._ns is not None else x
        result = FrontendResult(denoised=clean)
        p = self.params
        for prob, rms in self._vad.process(clean):
            voiced = rms >= self._min_rms
            if not self.speaking:
                self._run_ms = self._run_ms + WINDOW_MS if prob >= p.threshold and voiced else 0
                if self._run_ms >= p.start_ms:
                    self.speaking = True
                    self._speech_ms = self._run_ms
                    self._run_ms = 0
                    result.events.append(VadEvent("start"))
            else:
                self._speech_ms += WINDOW_MS
                self._run_ms = self._run_ms + WINDOW_MS if prob < p.stop_threshold or not voiced else 0
                if self._run_ms == 0:
                    if self._paused:
                        result.events.append(VadEvent("resume"))
                    self._paused = False
                elif not self._paused and self._run_ms >= p.pause_ms:
                    self._paused = True
                    result.events.append(VadEvent("pause", self._speech_ms - self._run_ms))
                if self._run_ms >= p.stop_ms:
                    self._paused = False
                    self.speaking = False
                    result.events.append(VadEvent("stop", self._speech_ms - self._run_ms))
                    self._run_ms = 0
                    self._speech_ms = 0
        result.speaking = self.speaking
        result.speech_ms = self._speech_ms if self.speaking else 0
        return result

    def close(self) -> None:
        if self._ns is not None:
            self._ns.close()


def float_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
    return 20 * math.log10(rms) if rms > 0 else -120.0
