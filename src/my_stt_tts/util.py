"""Small shared utilities."""

from __future__ import annotations

import io
import time
import wave
from collections import deque
from collections.abc import Callable

import numpy as np


def wav_bytes_from_int16(pcm16: bytes, sample_rate: int) -> bytes:
    """Wrap headerless little-endian int16 mono PCM in an in-memory WAV container.

    Shared by the cloud adapters (G1) so the WAV-header boilerplate lives in one
    place. ``pcm16`` is the raw int16-LE sample bytes.
    """
    buf = io.BytesIO()
    # pylint: disable=no-member  # wave.open(..., "wb") -> Wave_write
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16)
    return buf.getvalue()


def wav_bytes_from_float(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono PCM in [-1, 1] as an in-memory 16-bit WAV byte string."""
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return wav_bytes_from_int16((pcm * 32767.0).astype("<i2").tobytes(), sample_rate)


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
