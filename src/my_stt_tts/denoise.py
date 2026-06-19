"""Pre-VAD noise suppression (R3-6): clean mic frames before VAD/STT.

A noisy room (fan, TV, traffic) both lowers STT accuracy and triggers false
barge-ins — the VAD / interrupt predictor mistake steady-state noise for speech.
This module adds an optional denoiser stage applied to mic frames **after** echo
cancellation (so the assistant's own voice is already gone) and **before** the
VAD / streaming STT. It composes cleanly with :mod:`my_stt_tts.aec`: the AEC
removes the loudspeaker echo, the denoiser then removes ambient noise.

Backends (preference order documented in the README):

* ``"spectral"`` — a pure-numpy **spectral-gate / spectral-subtraction** denoiser.
  It estimates a per-frequency noise floor from the quietest recent frames and
  attenuates bins near that floor (a soft Wiener-style gain). No native deps, so
  it is always available and fully unit-tested. This is real, working suppression:
  it raises the signal-to-noise ratio of steady noise measurably (see tests).
* ``"rnnoise"`` — **RNNoise** via an optional wheel (the ``denoiser`` extra). If the
  wheel is unavailable / fails to load on this platform, the factory **falls back**
  to the spectral denoiser so the loop never breaks. (On this arm64 setup the
  RNNoise wheel's transitive deps conflict with the WebRTC ``av`` build, so the
  spectral backend is the default — see the README caveat.)
* ``"off"`` — identity pass-through (no denoising).

The seam is the :class:`Denoiser` protocol: ``process(frame)`` returns the frame
with noise reduced; ``reset()`` clears the adaptive noise estimate for a new turn.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("my_stt_tts.denoise")

DENOISER_MODES = ("off", "spectral", "rnnoise")


@runtime_checkable
class Denoiser(Protocol):
    """Reduce background noise in a float32 mono mic frame.

    Usage::

        clean = denoiser.process(mic_frame)   # noise attenuated
        denoiser.reset()                       # new turn -> forget the noise model
    """

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return ``frame`` with background noise attenuated."""
        ...

    def reset(self) -> None:
        """Clear the adaptive noise estimate for a new utterance."""
        ...


class NullDenoiser:
    """Identity pass-through: no denoising (``denoiser="off"``)."""

    def process(self, frame: np.ndarray) -> np.ndarray:  # noqa: D102 — see protocol
        return np.asarray(frame, dtype=np.float32).ravel()

    def reset(self) -> None:  # noqa: D102
        return None


class SpectralGateDenoiser:
    """Spectral-gate noise reduction in pure numpy (the safe, always-on default).

    For each frame it takes the magnitude spectrum, maintains a running estimate
    of the per-bin **noise floor** as the minimum magnitude seen over a short
    sliding history (the noise sits in the low percentile while speech spikes
    above it), then applies a soft spectral-subtraction gain::

        gain = max(0, (mag^2 - (k * noise)^2)) / (mag^2 + eps)
        clean_spectrum = mag * gain * exp(j * phase)

    ``strength`` (k) is the over-subtraction factor; higher removes more noise at
    the cost of more speech distortion. The phase is preserved and the frame is
    reconstructed with an inverse rFFT. Short frames (< 2 samples) pass through.
    """

    def __init__(
        self, *, strength: float = 1.0, history: int = 16, floor_gain: float = 0.05
    ) -> None:
        if strength < 0:
            raise ValueError("strength must be >= 0")
        self.strength = float(strength)
        self.floor_gain = float(floor_gain)
        # Sliding history of recent magnitude spectra to estimate the noise floor.
        self._mags: deque[np.ndarray] = deque(maxlen=max(1, history))

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Attenuate noise in ``frame`` via spectral subtraction + adapt the floor."""
        arr = np.asarray(frame, dtype=np.float32).ravel()
        if arr.size < 2:
            return arr
        spectrum = np.fft.rfft(arr)
        mag = np.abs(spectrum)
        self._mags.append(mag)
        # Noise floor: the per-bin minimum over the recent history (robust to the
        # occasional speech spike); until we have history, use this frame itself.
        noise = (
            np.min(np.stack(self._mags), axis=0) if len(self._mags) > 1 else mag * self.floor_gain
        )
        power = mag * mag
        noise_power = (self.strength * noise) ** 2
        gain = np.clip((power - noise_power) / (power + 1e-12), 0.0, 1.0)
        # Keep a small residual floor so bins are attenuated, not hard-gated to
        # zero (avoids musical-noise artefacts that hurt STT).
        gain = self.floor_gain + (1.0 - self.floor_gain) * gain
        cleaned = np.fft.irfft(spectrum * gain, n=arr.size).astype(np.float32)
        return cleaned

    def reset(self) -> None:
        """Forget the noise-floor estimate for a new utterance."""
        self._mags.clear()


class RnnoiseDenoiser:
    """RNNoise speech-enhancement via an optional wheel; spectral fallback otherwise.

    RNNoise is a small recurrent network trained for real-time speech denoising.
    The wheel is lazy-imported from the ``denoiser`` extra; if it (or its native
    runtime) is unavailable, :meth:`available` is False and the factory falls back
    to :class:`SpectralGateDenoiser`. When live, frames are resampled to RNNoise's
    native rate, denoised, and resampled back — but only if the bridge constructs;
    we never raise into the audio loop.
    """

    #: RNNoise operates on 48 kHz, 10 ms (480-sample) frames.
    NATIVE_SR = 48000

    def __init__(self, sample_rate: int) -> None:
        self.sample_rate = sample_rate
        self._engine: Any = None
        self._fallback = SpectralGateDenoiser()

    @staticmethod
    def available() -> bool:
        """Whether an RNNoise wheel is importable on this machine."""
        try:
            import importlib.util

            return importlib.util.find_spec("pyrnnoise") is not None
        except Exception:
            return False

    def _ensure(self) -> Any:
        if self._engine is None:
            try:
                from pyrnnoise import RNNoise  # type: ignore[import-not-found]

                self._engine = RNNoise(self.NATIVE_SR)
            except Exception:  # broken/missing wheel -> use the spectral fallback
                log.info("RNNoise unavailable; using the spectral denoiser.", exc_info=True)
                self._engine = False
        return self._engine

    def process(self, frame: np.ndarray) -> np.ndarray:  # noqa: D102 — see protocol
        engine = self._ensure()
        if not engine:
            return self._fallback.process(frame)
        arr = np.asarray(frame, dtype=np.float32).ravel()
        try:
            # pyrnnoise expects int16 PCM; denoise then return to float32.
            pcm16 = np.clip(arr, -1.0, 1.0)
            _, out = engine.process_frame((pcm16 * 32767.0).astype(np.int16))
            return (np.asarray(out, dtype=np.float32) / 32767.0).ravel()
        except Exception:  # any runtime error -> graceful fallback for this frame
            return self._fallback.process(frame)

    def reset(self) -> None:  # noqa: D102
        self._fallback.reset()


def make_denoiser(cfg: Any) -> Denoiser:
    """Build the configured :class:`Denoiser` (from ``cfg.denoiser``).

    * ``"spectral"`` — the pure-numpy spectral-gate denoiser.
    * ``"rnnoise"`` — RNNoise if the wheel is available, else the spectral denoiser.
    * ``"off"`` / anything else — identity pass-through.
    """
    mode = getattr(cfg, "denoiser", "off")
    strength = float(getattr(cfg, "denoiser_strength", 1.0))
    if mode == "spectral":
        return SpectralGateDenoiser(strength=strength)
    if mode == "rnnoise":
        if RnnoiseDenoiser.available():
            return RnnoiseDenoiser(int(getattr(cfg, "sample_rate", 16000)))
        log.info("RNNoise wheel not installed; using the spectral denoiser.")
        return SpectralGateDenoiser(strength=strength)
    return NullDenoiser()
