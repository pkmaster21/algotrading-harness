"""A sliding-window rate limiter shared by everything that calls Alpaca."""

from __future__ import annotations

import time
from collections.abc import Callable


class RateLimiter:
    """Allows at most `max_per_minute` request tokens in any 60-second window."""

    def __init__(
        self,
        max_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._max = max_per_minute
        self._clock = clock
        self._sleep = sleep
        self._stamps: list[float] = []

    def acquire(self, n: int = 1) -> None:
        for _ in range(n):
            while True:
                now = self._clock()
                self._stamps = [t for t in self._stamps if now - t < 60.0]
                if len(self._stamps) < self._max:
                    self._stamps.append(now)
                    break
                self._sleep(60.0 - (now - self._stamps[0]))
