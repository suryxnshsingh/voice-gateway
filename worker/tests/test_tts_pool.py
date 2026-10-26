"""TTSPool bookkeeping with a fake model: instances always come back, and a leak heals itself."""

import asyncio
import contextlib
import time
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

import numpy as np
import pytest
import torch

from vgworker.tts import StreamStats, TTSBusy, TTSPool


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
