"""Small shared utilities."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable


class RateLimiter:
    """Sliding-window limiter: at most ``per_minute`` acquisitions per 60 s.

    Guards against runaway loops (e.g. a self-trigger firing the LLM repeatedly).
    ``clock`` is injectable for testing.
    """

    def __init__(self, per_minute: int, clock: Callable[[], float] = time.monotonic) -> None:
        self.capacity = per_minute
        self._clock = clock
        self._events: deque[float] = deque()

    def acquire(self) -> bool:
        """Record an event if under the limit; return ``False`` if rate-exceeded."""
        now = self._clock()
        while self._events and now - self._events[0] >= 60.0:
            self._events.popleft()
        if len(self._events) >= self.capacity:
            return False
        self._events.append(now)
        return True
