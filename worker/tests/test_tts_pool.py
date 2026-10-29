"""TTSPool bookkeeping with a fake model: instances always come back, and a leak heals itself."""

import asyncio
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import numpy as np
import pytest
import torch

from vgworker.tts import AdapterNotFound, StreamStats, TTSBusy, TTSPool


class FakeModel:
    sample_rate = 48000

    def build_prompt_cache(self, **kw):
        return {"mode": "reference"}

    def generate_with_prompt_cache_streaming(self, target_text, prompt_cache, **kw):
        for _ in range(5):
            time.sleep(0.02)
            yield torch.zeros(1, 7680), None, None  # 160 ms at 48 kHz


def make_pool(tmp_path, n=1) -> TTSPool:
    pool = TTSPool.__new__(TTSPool)  # skip loading VoxCPM
    pool._models = [FakeModel() for _ in range(n)]
    pool.sample_rate, pool.device = 48000, "cpu"
    pool._idle = asyncio.Queue()
    for m in pool._models:
        pool._idle.put_nowait(m)
    pool._holders = {}
    pool._executor = ThreadPoolExecutor(max_workers=n)
    pool._voices_dir = tmp_path
    pool._cfg, pool._timesteps = 2.0, 10
    pool._caches, pool._cache_size, pool._loading = OrderedDict(), 8, {}
    pool.last_used = time.monotonic()
    pool._caches["v"] = {"mode": "reference"}
    pool._init_adapters(tmp_path / "adapters", False, 4)
    return pool


async def settle(pool, want_idle, timeout=2.0):
    t0 = time.monotonic()
    while pool.idle != want_idle and time.monotonic() - t0 < timeout:
        await asyncio.sleep(0.01)
    return pool.idle


async def test_instance_returns_after_full_stream(tmp_path):
    pool = make_pool(tmp_path)
    async with contextlib.aclosing(pool.stream("v", "hi", 24000, StreamStats())) as chunks:
        n = len([c async for c in chunks])
    assert n >= 5 and await settle(pool, 1) == 1 and pool.busy() == []


async def test_instance_returns_when_client_stops_early(tmp_path):
    pool = make_pool(tmp_path)
    async with contextlib.aclosing(pool.stream("v", "hi", 24000, StreamStats())) as chunks:
        async for _ in chunks:
            break
    assert await settle(pool, 1) == 1


async def test_queue_timeout_raises_instead_of_hanging(tmp_path):
    pool = make_pool(tmp_path)
    await pool._acquire("someone else", timeout=1)
    with pytest.raises(TTSBusy):
        async with contextlib.aclosing(pool.stream("v", "hi", 24000, StreamStats(), queue_timeout=0.1)) as chunks:
            async for _ in chunks:
                pass


async def test_leaked_instance_is_reclaimed_once(tmp_path):
    pool = make_pool(tmp_path)
    model = await pool._acquire("leaky op", timeout=1)
    holder = pool._holders[id(model)]
    done = asyncio.get_running_loop().create_future()
    done.set_result(None)
    pool._track(model, done)  # thread finished, but nobody hands the instance back
    holder.since -= 10
    assert pool.reclaim_leaked() == ["leaky op"] and pool.idle == 1
    # Someone else takes it; a late release from the leaky holder must not free it from under them.
    again = await pool._acquire("new owner", timeout=1)
    pool._release(model, holder)
    assert pool.idle == 0 and pool.busy()[0]["op"] == "new owner"
    pool._release(again, pool._holders[id(again)])
    assert pool.idle == 1


async def test_keep_warm_only_when_idle_and_returns_the_instance(tmp_path):
    pool = make_pool(tmp_path)
    pool.last_used = 0
    assert await pool.keep_warm() is True and pool.idle == 1 and pool.busy() == []
    held = await pool._acquire("someone", timeout=1)
    assert await pool.keep_warm() is False  # never competes with a real request
    pool._release(held, pool._holders[id(held)])


class LoraModel(FakeModel):
    """Two LoRA tensors, the way a compiled VoxCPM2 names them."""

    def __init__(self):
        self.params = {"lm._orig_mod.q_proj.lora_A": torch.nn.Parameter(torch.ones(2, 2)),
                       "lm._orig_mod.q_proj.lora_B": torch.nn.Parameter(torch.ones(2, 2))}
        self.resets = 0

    def named_parameters(self):
        return iter(self.params.items())

    def reset_lora_weights(self):
        self.resets += 1
        self.params["lm._orig_mod.q_proj.lora_B"].data.zero_()


def adapter_pool(tmp_path):
    from safetensors.torch import save_file

    pool = make_pool(tmp_path)
    pool._models = [LoraModel()]
    pool._idle = asyncio.Queue()
    pool._idle.put_nowait(pool._models[0])
    pool._init_adapters(tmp_path / "adapters", True, 4)
    (tmp_path / "adapters" / "teacher").mkdir(parents=True)
    save_file({"lm.q_proj.lora_A": torch.full((2, 2), 3.0), "lm.q_proj.lora_B": torch.full((2, 2), 5.0)},
              str(tmp_path / "adapters" / "teacher" / "lora_weights.safetensors"))
    return pool


async def say(pool, adapter=None):
    async with contextlib.aclosing(pool.stream("v", "hi", 24000, StreamStats(), adapter=adapter)) as chunks:
        return len([c async for c in chunks])


async def test_a_fine_tuned_voice_gets_its_adapter_and_others_get_the_base(tmp_path):
    pool = adapter_pool(tmp_path)
    model = pool._models[0]
    b = model.params["lm._orig_mod.q_proj.lora_B"]
    assert await say(pool, "teacher") >= 5
    assert float(b.data[0, 0]) == 5.0 and pool.adapters_loaded == 1
    await say(pool)  # a cloned voice: LoRA zeroed, so exactly the base model
    assert float(b.data.abs().sum()) == 0.0
    resets = model.resets
    await say(pool)
    assert model.resets == resets  # already on the base: nothing to copy
    await say(pool, "teacher")
    assert float(b.data[0, 0]) == 5.0 and await settle(pool, 1) == 1


async def test_a_missing_adapter_fails_fast_and_frees_nothing(tmp_path):
    pool = adapter_pool(tmp_path)
    with pytest.raises(AdapterNotFound):
        await say(pool, "nobody")
    assert pool.idle == 1 and pool.busy() == []


async def test_an_adapter_that_does_not_fit_is_refused(tmp_path):
    from safetensors.torch import save_file

    pool = adapter_pool(tmp_path)
    (tmp_path / "adapters" / "other").mkdir()
    save_file({"dit.k_proj.lora_A": torch.ones(2, 2)}, str(tmp_path / "adapters" / "other" / "lora_weights.safetensors"))
    with pytest.raises(AdapterNotFound):
        await say(pool, "other")
    assert await settle(pool, 1) == 1
    await say(pool)  # the half switched instance is reset before the next voice
    assert float(pool._models[0].params["lm._orig_mod.q_proj.lora_B"].data.abs().sum()) == 0.0
