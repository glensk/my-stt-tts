"""Earcons (short chimes) used instead of spoken status cues."""

from __future__ import annotations

import numpy as np

DEFAULT_SR = 24000


def _fade(n: int, sample_rate: int, fade_s: float = 0.01) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    k = min(int(sample_rate * fade_s), n // 2)
    if k > 0:
        ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
        env[:k] = ramp
        env[-k:] = ramp[::-1]
    return env


def tone(
    freqs: list[float],
    *,
    duration: float = 0.18,
    sample_rate: int = DEFAULT_SR,
    volume: float = 0.3,
) -> np.ndarray:
    """Render a short multi-note earcon as a float32 mono array."""
    seg = duration / len(freqs)
    parts: list[np.ndarray] = []
    for freq in freqs:
        n = int(sample_rate * seg)
        t = np.linspace(0.0, seg, n, endpoint=False)
        wave = np.sin(2 * np.pi * freq * t).astype(np.float32)
        parts.append(wave * _fade(n, sample_rate))
    return (np.concatenate(parts) * volume).astype(np.float32)


def chime_listening(sample_rate: int = DEFAULT_SR) -> np.ndarray:
    """Rising two-note cue: mic is now live."""
    return tone([880.0, 1320.0], sample_rate=sample_rate)


def chime_done(sample_rate: int = DEFAULT_SR) -> np.ndarray:
    """Falling two-note cue: capture finished."""
    return tone([660.0, 440.0], sample_rate=sample_rate)


def chime_error(sample_rate: int = DEFAULT_SR) -> np.ndarray:
    """Descending three-note cue: something failed."""
    return tone([440.0, 330.0, 220.0], duration=0.3, sample_rate=sample_rate)
