"""Per-customer request limits: concurrency per feature and requests per minute.

In-process, so per gateway; with several gateways behind a load balancer, move these to Redis.
"""

import collections
import contextlib
import time

from fastapi import HTTPException


class Limiter:
    def __init__(self, per_minute: int, concurrency: dict[str, int]):
        self._per_minute = per_minute
        self._concurrency = concurrency
        self._hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._active: dict[tuple[str, str], int] = collections.defaultdict(int)

    def check_rate(self, customer_id: str) -> None:
        now = time.monotonic()
        hits = self._hits[customer_id]
        while hits and now - hits[0] > 60:
            hits.popleft()
        if len(hits) >= self._per_minute:
            raise HTTPException(429, "Too many requests. Please slow down.", headers={"Retry-After": "5"})
        hits.append(now)

    def acquire(self, customer_id: str, feature: str) -> tuple[str, str]:
        """Raises 429 if over the rate or concurrency limit. Pair with release() (streams hold it to the end)."""
        self.check_rate(customer_id)
        key = (customer_id, feature)
        if self._active[key] >= self._concurrency.get(feature, 4):
            raise HTTPException(429, f"Too many concurrent {feature} requests.", headers={"Retry-After": "2"})
        self._active[key] += 1
        return key

    def release(self, key: tuple[str, str]) -> None:
        self._active[key] = max(0, self._active[key] - 1)

    @contextlib.asynccontextmanager
    async def slot(self, customer_id: str, feature: str):
        key = self.acquire(customer_id, feature)
        try:
            yield
        finally:
            self.release(key)
