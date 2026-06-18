"""False-interrupt suppression for barge-in (Phase 7, G4).

Without acoustic echo cancellation (AEC) the mic hears the speakers, and even with
headphones a cough, a "mhm" backchannel, or TV noise should NOT abort the
assistant mid-sentence. :class:`InterruptGate` is a small, pure state machine that
decides whether detected user speech is a *genuine* interruption.

Two pipecat-style guards (``MinWordsUserTurnStartStrategy`` equivalent):

* **minimum speech duration** — accumulate contiguous voiced milliseconds; a
  short blip never trips the gate;
* **minimum word count** — once partial-STT text is available, require at least
  N words.

Either guard alone can authorise an interruption (``min_words`` defaults low, so
duration usually fires first), so the gate works even before partial transcripts
exist. An optional per-frame **energy floor** is applied by the caller (see
:func:`frame_energy`) to reject low-level open-speaker bleed before a frame is
ever fed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_WORD = re.compile(r"\b\w[\w'-]*\b", re.UNICODE)


def word_count(text: str) -> int:
    """Count word-like tokens in ``text`` (unicode-aware; ignores punctuation)."""
    return len(_WORD.findall(text or ""))


def frame_energy(frame: Any) -> float:
    """Root-mean-square energy of a float32 mono frame (0..~1).

    Used as a cheap gate against open-speaker playback bleed: a frame below the
    configured ``barge_in_energy`` floor is treated as silence for barge-in.
    """
    arr = np.asarray(frame, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(arr * arr)))


@dataclass
class InterruptGate:
    """Decide whether user speech during playback is a real interruption.

    Feed each mic frame's outcome via :meth:`update`. The gate "opens" (returns
    ``True`` once, then stays open until :meth:`reset`) when EITHER enough voiced
    time OR enough words have accumulated. ``min_words == 0`` disables the
    word guard; ``min_speech_ms == 0`` disables the duration guard.
    """

    min_speech_ms: float
    min_words: int
    frame_ms: float
    _voiced_ms: float = field(default=0.0, init=False)
    _words: int = field(default=0, init=False)
    _open: bool = field(default=False, init=False)

    def update(self, is_speech: bool, *, partial_text: str | None = None) -> bool:
        """Feed one frame. ``is_speech`` is the VAD/energy verdict for the frame;
        ``partial_text`` is the latest partial transcript if streaming STT is on.

        Returns ``True`` the moment the interruption is authorised (and on every
        subsequent call until :meth:`reset`)."""
        if is_speech:
            self._voiced_ms += self.frame_ms
        if partial_text is not None:
            self._words = word_count(partial_text)
        if self._open or self._satisfied():
            self._open = True
        return self._open

    def _satisfied(self) -> bool:
        # Require that *some* speech has been heard at all; then either guard.
        if self._voiced_ms <= 0.0:
            return False
        duration_ok = self.min_speech_ms > 0.0 and self._voiced_ms >= self.min_speech_ms
        words_ok = self.min_words > 0 and self._words >= self.min_words
        # If both guards are disabled, any voiced frame interrupts.
        if self.min_speech_ms <= 0.0 and self.min_words <= 0:
            return True
        return duration_ok or words_ok

    @property
    def open(self) -> bool:
        """Whether the gate has already authorised an interruption this turn."""
        return self._open

    @property
    def voiced_ms(self) -> float:
        """Contiguous voiced milliseconds accumulated so far."""
        return self._voiced_ms

    def reset(self) -> None:
        """Clear all state for a new playback turn."""
        self._voiced_ms = 0.0
        self._words = 0
        self._open = False


def _spectral_flux(frame: np.ndarray, prev_mag: np.ndarray | None) -> tuple[float, np.ndarray]:
    """Half-wave-rectified spectral flux between successive frames + the new spectrum.

    Flux is the summed positive change in magnitude spectrum from one frame to the
    next. Sustained, evolving *speech* keeps flux moderate-to-high; steady tones
    (a TV hum) or silence keep it near zero. Returned normalised by spectrum size.
    """
    mag = np.abs(np.fft.rfft(frame * np.hanning(frame.size))) if frame.size > 1 else np.zeros(1)
    if prev_mag is None or prev_mag.shape != mag.shape:
        return 0.0, mag
    diff = mag - prev_mag
    flux = float(np.sum(diff[diff > 0]))
    return flux / max(1, mag.size), mag


def _zero_crossing_rate(frame: np.ndarray) -> float:
    """Fraction of sign changes in ``frame`` (0..1). Voiced speech is low-to-mid;
    fricatives/noise are high. Used only as a weak de-emphasis on noise bursts."""
    if frame.size < 2:
        return 0.0
    signs = np.signbit(frame)
    return float(np.count_nonzero(signs[1:] != signs[:-1])) / (frame.size - 1)


@dataclass
class InterruptPredictor:
    """Acoustic "intent-to-take-the-floor" guard for barge-in (R2-3).

    A principled, on-device score over the barge-in audio that distinguishes a
    *sustained real interruption* from a backchannel ("mhm"), a cough, or TV/room
    noise — without waiting for two words to transcribe. It accumulates a
    confidence score from three cues, frame by frame:

    * **sustained voiced energy** — voiced frames whose RMS clears ``energy_floor``
      build the score; below it, the score decays (so a single blip can't win);
    * **duration** — the score can only *fire* after ``min_ms`` of voiced audio,
      so transients (coughs, clicks) are rejected even if momentarily loud;
    * **spectral flux + ZCR** — evolving, low-ZCR spectra (real speech) are rewarded;
      flat (steady tone) or very high-ZCR (hiss/fricative-only) frames are damped.

    :meth:`update` returns ``True`` once the accumulated score crosses ``threshold``
    *and* the duration floor is met; it then stays open until :meth:`reset`. It is
    composed with :class:`InterruptGate` by the monitor loop (either may fire), so
    the assistant talks through backchannels but yields to a real interruption fast.
    Pure (no I/O), so it is unit-tested with synthetic frames.
    """

    threshold: float
    frame_ms: float
    min_ms: float = 240.0
    energy_floor: float = 0.02
    attack: float = 0.34  # score gained per strongly-voiced frame
    release: float = 0.25  # score lost per non-voiced frame
    _score: float = field(default=0.0, init=False)
    _voiced_ms: float = field(default=0.0, init=False)
    _prev_mag: np.ndarray | None = field(default=None, init=False)
    _open: bool = field(default=False, init=False)

    def update(self, frame: Any, is_speech: bool) -> bool:
        """Feed one (echo-cancelled) mic frame + its VAD verdict; return the verdict.

        Returns ``True`` the moment intent-to-take-the-floor is detected (and on
        every later call until :meth:`reset`)."""
        arr = np.asarray(frame, dtype=np.float32).ravel()
        energy = frame_energy(arr)
        flux, self._prev_mag = _spectral_flux(arr, self._prev_mag)
        zcr = _zero_crossing_rate(arr)
        voiced = is_speech and energy >= self.energy_floor
        if voiced:
            self._voiced_ms += self.frame_ms
            # Reward sustained voiced energy; modulate by how "speech-like" the
            # spectrum is (evolving spectrum, not pure hiss).
            flux_gain = min(1.0, flux * 64.0)  # normalise typical speech flux -> ~1
            zcr_damp = 1.0 - min(0.6, max(0.0, zcr - 0.35))  # damp high-ZCR noise
            self._score += self.attack * (0.5 + 0.5 * flux_gain) * zcr_damp
        else:
            self._voiced_ms = max(0.0, self._voiced_ms - self.frame_ms)
            self._score -= self.release
        self._score = min(1.0, max(0.0, self._score))
        if self._open or (self._score >= self.threshold and self._voiced_ms >= self.min_ms):
            self._open = True
        return self._open

    @property
    def score(self) -> float:
        """Current accumulated intent score in [0, 1]."""
        return self._score

    @property
    def open(self) -> bool:
        """Whether intent-to-take-the-floor has already been detected this turn."""
        return self._open

    def reset(self) -> None:
        """Clear all state for a new playback turn."""
        self._score = 0.0
        self._voiced_ms = 0.0
        self._prev_mag = None
        self._open = False


def make_interrupt_predictor(cfg: Any, frame_ms: float) -> InterruptPredictor | None:
    """Build the configured :class:`InterruptPredictor`, or None when disabled."""
    if not getattr(cfg, "interrupt_predict", False):
        return None
    return InterruptPredictor(
        threshold=float(getattr(cfg, "interrupt_predict_threshold", 0.6)),
        frame_ms=frame_ms,
        min_ms=float(getattr(cfg, "interrupt_predict_min_ms", 240.0)),
        energy_floor=float(getattr(cfg, "barge_in_energy", 0.02)),
    )
