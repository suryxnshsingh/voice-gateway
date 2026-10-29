"""VoxCPM2 streaming TTS behind a pool of model instances.

Each instance owns its own KV cache, so it renders one sentence at a time. A request waits
for an idle instance, streams 160 ms chunks from a worker thread, and hands the instance back
when the sentence finishes or the caller stops listening.

Every voice runs on the same base weights. A cloned voice is only a prompt cache (its reference
audio, encoded). A fine-tuned voice also names a LoRA adapter, which is copied into the instance's
LoRA layers before its sentence renders; any other voice gets the layers zeroed, which is exactly
the base model. Values are copied in place rather than switching LoRA off, so the compiled
graph never changes and nothing recompiles mid request.
"""

import asyncio
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr

from .audio import float_to_pcm16

log = logging.getLogger(__name__)


@dataclass
class _Holder:
    op: str
    since: float
    model: object
    fut: asyncio.Future | None = None


class VoiceNotFound(Exception):
    pass


class AdapterNotFound(Exception):
    """The voice names an adapter this worker doesn't have, or one that doesn't fit the model."""


class TTSBusy(Exception):
    """No instance freed up in time. Better a visible error than a reply that never speaks."""


@dataclass
class StreamStats:
    queue_ms: float = 0.0
    first_chunk_ms: float | None = None
    total_ms: float = 0.0
    audio_ms: float = 0.0


class TTSPool:
    def __init__(
        self,
        model_id: str,
        instances: int,
        device: str,
        voices_dir: Path,
        cfg_value: float,
        timesteps: int,
        cache_size: int = 256,
        adapters_dir: Path | None = None,
        lora_r: int = 0,
        lora_alpha: int = 0,
        adapter_cache: int = 16,
    ):
        import torch
        from voxcpm import VoxCPM

        dev = None if device == "auto" else device
        # torch.compile only works on CUDA; the first sentence per instance pays the compile.
        optimize = device.startswith("cuda") or (device == "auto" and torch.cuda.is_available())
        lora = None
        if lora_r > 0:
            from voxcpm.model.voxcpm2 import LoRAConfig

            lora = LoRAConfig(enable_lm=True, enable_dit=True, enable_proj=False, r=lora_r, alpha=lora_alpha)
        self._models = []
        for i in range(instances):
            m = VoxCPM.from_pretrained(model_id, load_denoiser=False, optimize=optimize, device=dev, lora_config=lora)
            if lora is not None:
                m.tts_model.reset_lora_weights()  # B = 0: the base model until an adapter is copied in
            self._models.append(m.tts_model)
            log.info("tts instance %d ready on %s", i, m.tts_model.device)
        self._init_adapters(adapters_dir or voices_dir.parent / "adapters", lora is not None, adapter_cache)
        self.sample_rate: int = self._models[0].sample_rate
        self.device: str = str(self._models[0].device)
        self._idle: asyncio.Queue = asyncio.Queue()
        for m in self._models:
            self._idle.put_nowait(m)
        # id(model) -> who has it. Lets /health and the watchdog say what is stuck, and lets the
        # watchdog take back an instance whose render thread has finished but was never returned.
        self._holders: dict[int, _Holder] = {}
        self._executor = ThreadPoolExecutor(max_workers=instances, thread_name_prefix="tts")
        self._voices_dir = voices_dir
        self._cfg = cfg_value
        self._timesteps = timesteps
        # Prompt caches are CPU tensors (VoxCPM moves encoded features to CPU), so every instance can share them.
        self._caches: OrderedDict[str, dict] = OrderedDict()
        self._cache_size = cache_size
        self._loading: dict[str, asyncio.Task] = {}
        self.last_used = time.monotonic()

    def _init_adapters(self, adapters_dir: Path, enabled: bool, cache: int) -> None:
        self._lora = enabled
        self._adapters_dir = adapters_dir
        self._adapter_cache = cache
        self._adapters: OrderedDict[str, dict] = OrderedDict()  # name -> LoRA tensors on the CPU
        self._adapter_lock = threading.Lock()  # render threads of several instances share the cache
        self._active: dict[int, object] = {}  # id(model) -> adapter name in its LoRA layers (None = base)
        self._lora_params: dict[int, dict] = {}

    def adapter_file(self, name: str) -> Path | None:
        for path in (self._adapters_dir / name / "lora_weights.safetensors", self._adapters_dir / f"{name}.safetensors"):
            if path.exists():
                return path
        return None

    @property
    def adapters_loaded(self) -> int:
        return len(self._adapters)

    def _adapter_state(self, name: str) -> dict:
        with self._adapter_lock:
            state = self._adapters.get(name)
            if state is not None:
                self._adapters.move_to_end(name)
                return state
        path = self.adapter_file(name)
        if path is None:
            raise AdapterNotFound(name)
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
        with self._adapter_lock:
            self._adapters[name] = state
            while len(self._adapters) > self._adapter_cache:
                self._adapters.popitem(last=False)
        return state

    def _use_adapter(self, model, name: str | None) -> None:
        """Put `name`'s LoRA weights (or zeros, for None) into the instance. Runs on its render thread
        while the instance is held, so nothing else is using it."""
        if self._active.get(id(model)) == name:
            return
        if not self._lora:
            if name is not None:
                raise AdapterNotFound(name)
            return
        state = self._adapter_state(name) if name is not None else None
        self._active[id(model)] = object()  # half switched until the copy below finishes
        model.reset_lora_weights()
        if state is not None:
            params = self._lora_params.get(id(model))
            if params is None:
                # torch.compile wraps modules as _orig_mod; adapters are saved without it.
                params = {k.replace("._orig_mod.", "."): p for k, p in model.named_parameters() if "lora_" in k}
                self._lora_params[id(model)] = params
            missing = [k for k in state if k not in params]
            if missing:
                raise AdapterNotFound(f"{name} does not fit this model ({len(missing)} unknown tensors, e.g. {missing[0]})")
            for key, value in state.items():
                params[key].data.copy_(value)
        self._active[id(model)] = name

    async def keep_warm(self) -> bool:
        """Render one tiny word if every instance is idle. Under memory pressure (a Mac) an idle model gets
        swapped out and the next request pays seconds to page it back in; touching it keeps it resident."""
        if self.idle < self.instances:
            return False
        model = await self._acquire("keep-warm", timeout=0.01)
        holder = self._holders[id(model)]
        fut = asyncio.get_running_loop().run_in_executor(self._executor, lambda: [
            None for _ in model.generate_with_prompt_cache_streaming(target_text="हाँ।", prompt_cache=None, inference_timesteps=2)])
        self._track(model, fut)
        try:
            await asyncio.shield(fut)
            self._release(model, holder)
        finally:
            fut.add_done_callback(lambda _: self._release(model, holder))  # if we were cancelled mid-render
        return True

    async def warmup(self) -> None:
        """The first sentence per instance is ~5x slower (kernel setup, compile on CUDA); pay it at startup."""
        loop = asyncio.get_running_loop()

        def run(model) -> None:
            for _ in model.generate_with_prompt_cache_streaming(target_text="Warming up.", prompt_cache=None):
                pass

        await asyncio.gather(*(loop.run_in_executor(self._executor, run, m) for m in self._models))

    async def _acquire(self, op: str, timeout: float):
        try:
            model = await asyncio.wait_for(self._idle.get(), timeout)
        except asyncio.TimeoutError:
            raise TTSBusy(f"no TTS instance free after {timeout:.0f} s; held by {self.busy()}") from None
        self._holders[id(model)] = _Holder(op, time.monotonic(), model)
        return model

    def _track(self, model, fut) -> None:
        if id(model) in self._holders:
            self._holders[id(model)].fut = fut

    def _release(self, model, holder: "_Holder | None" = None) -> None:
        """Hand an instance back once. With `holder`, only if that holder still has it, so a late
        release after a watchdog reclaim can't free an instance someone else now holds."""
        current = self._holders.get(id(model))
        if current is None or (holder is not None and current is not holder):
            return
        del self._holders[id(model)]
        self._idle.put_nowait(model)

    def busy(self) -> list[dict]:
        now = time.monotonic()
        return [{"op": h.op, "for_s": round(now - h.since, 1), "thread_done": h.fut is not None and h.fut.done()}
                for h in self._holders.values()]

    def reclaim_leaked(self, grace_s: float = 5.0) -> list[str]:
        """Return instances whose render thread is finished but which nobody handed back.

        Safe because the thread no longer touches the model; without this, one leak silences TTS
        for every session until a restart.
        """
        now = time.monotonic()
        leaked = [h for h in self._holders.values() if h.fut is not None and h.fut.done() and now - h.since > grace_s]
        for h in leaked:
            self._release(h.model, h)
        return [h.op for h in leaked]

    @property
    def instances(self) -> int:
        return len(self._models)

    @property
    def idle(self) -> int:
        return self._idle.qsize()

    def voice_path(self, voice_id: str) -> Path:
        return self._voices_dir / f"{voice_id}.wav"

    async def load_voice(self, voice_id: str) -> None:
        """Build (or rebuild) the prompt cache for a stored reference file."""
        path = self.voice_path(voice_id)
        if not path.exists():
            raise VoiceNotFound(voice_id)
        model = await self._acquire(f"load_voice {voice_id}", timeout=120)
        holder = self._holders[id(model)]

        def build() -> dict:
            cache = model.build_prompt_cache(reference_wav_path=str(path))
            # A voice's first sentence is ~6x slower (kernels specialise on its prompt length, on MPS
            # and under torch.compile alike); render a throwaway one now instead of in a live reply.
            for _ in model.generate_with_prompt_cache_streaming(target_text="Hello there.", prompt_cache=cache):
                pass
            return cache

        fut = asyncio.get_running_loop().run_in_executor(self._executor, build)
        self._track(model, fut)
        try:
            cache = await asyncio.shield(fut)
        finally:
            # Hand the instance back only when the thread is done with it, even if we were cancelled.
            fut.add_done_callback(lambda _: self._release(model, holder))
        self._caches[voice_id] = cache
        self._caches.move_to_end(voice_id)
        while len(self._caches) > self._cache_size:
            self._caches.popitem(last=False)

    def evict(self, voice_id: str) -> None:
        self._caches.pop(voice_id, None)

    @property
    def cached_voices(self) -> int:
        return len(self._caches)

    def is_cached(self, voice_id: str) -> bool:
        return voice_id in self._caches

    async def ensure_loaded(self, voice_id: str) -> None:
        """Load a stored voice once, however many requests ask for it at the same time."""
        if voice_id in self._caches:
            return
        task = self._loading.get(voice_id)
        if task is None:
            task = asyncio.create_task(self.load_voice(voice_id))
            self._loading[voice_id] = task
            task.add_done_callback(lambda _: self._loading.pop(voice_id, None))
        await asyncio.shield(task)

    async def _cache_for(self, voice_id: str) -> dict:
        await self.ensure_loaded(voice_id)
        self._caches.move_to_end(voice_id)
        return self._caches[voice_id]

    async def stream(self, voice_id: str, text: str, out_sr: int, stats: StreamStats,
                     queue_timeout: float = 15.0, timesteps: int | None = None,
                     cfg_value: float | None = None, adapter: str | None = None) -> AsyncIterator[bytes]:
        """Yield PCM16 mono chunks at out_sr. Use with contextlib.aclosing so cancellation frees the instance.
        `adapter` names the LoRA adapter of a fine-tuned voice; None renders on the base model."""
        t0 = time.monotonic()
        self.last_used = time.monotonic()
        if adapter is not None and (not self._lora or self.adapter_file(adapter) is None):
            raise AdapterNotFound(adapter)
        cache = await self._cache_for(voice_id)
        model = await self._acquire(f"tts {voice_id} {text[:30]!r}", timeout=queue_timeout)
        holder = self._holders[id(model)]
        stats.queue_ms = (time.monotonic() - t0) * 1000
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        _done = object()

        def run() -> None:
            try:
                self._use_adapter(model, adapter)
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, e)
                loop.call_soon_threadsafe(q.put_nowait, _done)
                return
            gen = model.generate_with_prompt_cache_streaming(
                target_text=text, prompt_cache=cache, cfg_value=cfg_value or self._cfg,
                inference_timesteps=timesteps or self._timesteps,
            )
            try:
                for wav, _, _ in gen:
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(q.put_nowait, wav.squeeze(0).float().numpy())
            except Exception as e:  # surfaced to the async side
                loop.call_soon_threadsafe(q.put_nowait, e)
            finally:
                gen.close()
                loop.call_soon_threadsafe(q.put_nowait, _done)

        fut = loop.run_in_executor(self._executor, run)
        self._track(model, fut)
        resampler = soxr.ResampleStream(self.sample_rate, out_sr, 1, dtype="float32") if out_sr != self.sample_rate else None
        try:
            while True:
                item = await q.get()
                if item is _done:
                    break
                if isinstance(item, Exception):
                    raise item
                chunk: np.ndarray = item
                stats.audio_ms += 1000 * len(chunk) / self.sample_rate
                if resampler is not None:
                    chunk = resampler.resample_chunk(chunk)
                if stats.first_chunk_ms is None:
                    stats.first_chunk_ms = (time.monotonic() - t0) * 1000
                if len(chunk):
                    yield float_to_pcm16(chunk)
            if resampler is not None:
                tail = resampler.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                if len(tail):
                    yield float_to_pcm16(tail)
        finally:
            stop.set()
            # The instance goes back only after its thread stops touching the KV cache. A callback
            # rather than an await, so a cancelled caller can't leak the instance.
            fut.add_done_callback(lambda _: self._release(model, holder))
            stats.total_ms = (time.monotonic() - t0) * 1000
