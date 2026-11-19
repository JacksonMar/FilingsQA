"""A minimal client-side rate limiter (SEC policy: at most 10 requests per second)."""

import threading
import time
from collections.abc import Callable


class RateLimiter:
    """Enforces a minimum interval between consecutive calls to `wait()`.

    Clock and sleep are injectable so the limiter can be unit-tested without real waiting.
    """

    def __init__(
        self,
        max_per_second: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_per_second <= 0:
            raise ValueError("max_per_second must be positive")
        self._min_interval = 1.0 / max_per_second
        self._clock = clock
        self._sleep = sleep
        self._next_allowed = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if now < self._next_allowed:
                self._sleep(self._next_allowed - now)
                now = self._next_allowed
            self._next_allowed = now + self._min_interval
