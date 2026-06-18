"""Acoustic echo cancellation (AEC) front-end for open-speaker barge-in (R2-1).

Without AEC the mic hears the assistant's own TTS coming out of the speakers, so
:func:`~my_stt_tts.audio.monitor_during_playback` would treat the bot's voice as a
user interruption. AEC removes the played-back signal (the *reference*) from the
mic capture so only the user's voice is left for the VAD / interrupt gate.

Three backends, in the preference order the README documents:

* ``"voiceprocessing"`` — macOS **hardware** AEC via ``AVAudioEngine`` +
  ``AVAudioInputNode.setVoiceProcessingEnabled`` (the VoiceProcessingIO unit).
  This is what FaceTime uses; it cancels in the OS audio HAL, so by the time mic
  frames reach Python they are *already* echo-cancelled and our software filter
  is a no-op. We can't run the real hardware unit in a unit test, so this backend
  only *enables* the OS feature (its availability is probed without a mic) and
  otherwise passes audio through unchanged.
* ``"nlms"`` — a pure-numpy **normalized least-mean-squares** adaptive filter that
  references the playback buffer. Always available (no native deps), fully
  unit-tested, and the cross-platform fallback. This is real, working AEC: it
  adapts a FIR filter to predict the echo from the reference and subtracts it.
* ``"off"`` — identity pass-through (legacy half-duplex behaviour).

The seam is the :class:`EchoCanceller` protocol: ``push_reference(samples)`` feeds
the audio that is being *played*, and ``process(frame)`` returns the mic ``frame``
with the echo removed. The monitor loop pushes the active
:class:`~my_stt_tts.tts.Playback` reference and processes each mic frame before VAD.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("my_stt_tts.aec")

AEC_MODES = ("off", "nlms", "voiceprocessing", "auto")


@runtime_checkable
class EchoCanceller(Protocol):
    """Removes the played-back (reference) signal from captured mic audio.

    Usage::

        aec.push_reference(played_samples)   # what the speaker is emitting
        clean = aec.process(mic_frame)       # mic with the echo subtracted
    """

    #: When True, the played-back signal is already removed (hardware AEC), so the
    #: caller may relax its open-speaker energy floor.
    active: bool

    def push_reference(self, samples: np.ndarray) -> None:
        """Feed the audio currently being played (the echo source)."""
        ...

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return ``frame`` (a mic chunk) with the echo cancelled."""
        ...

    def reset(self) -> None:
        """Clear all adaptive state for a new utterance."""
        ...


class NullEchoCanceller:
    """Identity pass-through: no cancellation (``aec="off"``)."""

    active: bool = False

    def push_reference(self, samples: np.ndarray) -> None:  # noqa: D102 — see protocol
        return None

    def process(self, frame: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.asarray(frame, dtype=np.float32).ravel()

    def reset(self) -> None:  # noqa: D102
        return None


class NlmsEchoCanceller:
    """Normalized LMS adaptive FIR echo canceller (pure numpy; the safe default).

    Models the echo path as an FIR filter ``w`` of ``taps`` coefficients applied to
    the recent reference (loudspeaker) signal. For each mic sample it predicts the
    echo ``y_hat = w · x``, subtracts it to get the error ``e = mic - y_hat``, and
    adapts ``w`` toward cancelling that error::

        w += mu * e * x / (||x||^2 + eps)

    The NLMS normalisation by the reference power makes convergence robust to the
    loudspeaker volume. ``process`` is sample-accurate but vectorised per call;
    ``mu`` (step size) trades convergence speed for stability. With no reference
    pushed yet, it is a clean pass-through.
    """

    active: bool = True

    def __init__(
        self, *, taps: int = 256, mu: float = 0.3, eps: float = 1e-6, ref_max_seconds: float = 8.0
    ) -> None:
        if taps <= 0:
            raise ValueError("taps must be > 0")
        if not 0.0 < mu <= 2.0:
            raise ValueError("mu must be in (0, 2]")
        self.taps = taps
        self.mu = mu
        self.eps = eps
        self._w = np.zeros(taps, dtype=np.float32)
        # Sliding history of recent reference samples (newest last). Bounded so the
        # buffer can't grow without limit during a long utterance.
        self._ref_maxlen = max(taps, int(16000 * ref_max_seconds))
        self._ref: deque[float] = deque(maxlen=self._ref_maxlen)
        # Seed with `taps` zeros so the first mic sample has a full reference window.
        self._ref.extend([0.0] * taps)

    def push_reference(self, samples: np.ndarray) -> None:
        """Append played-back samples to the reference history."""
        arr = np.asarray(samples, dtype=np.float32).ravel()
        self._ref.extend(arr.tolist())

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Cancel the echo from ``frame`` using the buffered reference + adapt ``w``."""
        mic = np.asarray(frame, dtype=np.float32).ravel()
        if mic.size == 0:
            return mic
        ref = np.fromiter(self._ref, dtype=np.float32, count=len(self._ref))
        if ref.size < self.taps:
            return mic  # not enough reference yet -> pass through
        out = np.empty(mic.size, dtype=np.float32)
        w = self._w
        taps = self.taps
        # Align so the LAST `mic.size` reference windows line up with the mic samples
        # (assumes the reference is roughly time-aligned with capture).
        ref_window = ref[-(taps + mic.size - 1) :] if ref.size >= taps + mic.size - 1 else ref
        if ref_window.size < taps + mic.size - 1:
            pad = np.zeros(taps + mic.size - 1 - ref_window.size, dtype=np.float32)
            ref_window = np.concatenate([pad, ref_window])
        for i in range(mic.size):
            x = ref_window[i : i + taps][::-1]  # most-recent-first FIR input
            y_hat = float(np.dot(w, x))
            err = mic[i] - y_hat
            norm = float(np.dot(x, x)) + self.eps
            w += (self.mu * err / norm) * x
            out[i] = err
        self._w = w
        return out

    def reset(self) -> None:
        """Forget the adapted filter + reference for a new utterance."""
        self._w = np.zeros(self.taps, dtype=np.float32)
        self._ref.clear()
        self._ref.extend([0.0] * self.taps)


class VoiceProcessingEchoCanceller:
    """macOS hardware AEC via ``AVAudioEngine`` VoiceProcessingIO (the best path).

    Enabling voice processing on the input node turns on the same echo canceller +
    noise suppressor FaceTime uses, *inside* the audio HAL. Mic frames that reach
    Python are therefore already echo-cancelled, so :meth:`process` is a
    pass-through and :attr:`active` reports True only when the OS feature could be
    enabled. PyObjC (the ``aec`` extra) is lazy-imported; if it is unavailable the
    factory falls back to NLMS and this backend reports ``active=False``.

    We deliberately do not open a live audio unit here (capture still flows through
    ``sounddevice``); :meth:`available` just verifies the API exists so the factory
    can decide. Probing for the API does not need a mic, so it is unit-testable.
    """

    def __init__(self) -> None:
        self.active = self._enable()

    @staticmethod
    def available() -> bool:
        """Whether the VoiceProcessingIO API is importable on this machine."""
        try:
            import AVFoundation  # noqa: PLC0415  (lazy, native)
        except Exception:  # PyObjC not installed / not macOS
            return False
        return hasattr(AVFoundation, "AVAudioEngine")

    def _enable(self) -> bool:
        """Turn on hardware voice processing on a probe engine. Returns success."""
        try:
            import AVFoundation  # noqa: PLC0415

            # pylint: disable=no-member  # PyObjC populates AVAudioEngine dynamically
            engine = AVFoundation.AVAudioEngine.alloc().init()
            input_node = engine.inputNode()
            ok, err = input_node.setVoiceProcessingEnabled_error_(True, None)
            if not ok:
                log.info("VoiceProcessingIO could not be enabled: %s", err)
                return False
            return bool(input_node.isVoiceProcessingEnabled())
        except Exception:  # any PyObjC / CoreAudio failure -> not active
            log.info("macOS VoiceProcessingIO unavailable; AEC handled in software.")
            return False

    def push_reference(self, samples: np.ndarray) -> None:  # noqa: D102 — HW handles echo
        return None

    def process(self, frame: np.ndarray) -> np.ndarray:  # noqa: D102 — already cancelled
        return np.asarray(frame, dtype=np.float32).ravel()

    def reset(self) -> None:  # noqa: D102
        return None


def make_echo_canceller(cfg: Any) -> EchoCanceller:
    """Build the configured :class:`EchoCanceller` (from ``cfg.aec_mode``).

    * ``"voiceprocessing"`` — hardware AEC if available, else NLMS.
    * ``"auto"`` — hardware AEC if available, else NLMS (the recommended default).
    * ``"nlms"`` — always the software adaptive filter.
    * ``"off"`` / anything else — identity pass-through.
    """
    mode = getattr(cfg, "aec_mode", "off")
    if mode == "off":
        return NullEchoCanceller()
    if mode == "nlms":
        return _make_nlms(cfg)
    if mode in ("voiceprocessing", "auto"):
        if VoiceProcessingEchoCanceller.available():
            hw = VoiceProcessingEchoCanceller()
            if hw.active:
                return hw
            log.info("hardware AEC inactive; falling back to software NLMS.")
        elif mode == "voiceprocessing":
            log.info("VoiceProcessingIO unavailable (no PyObjC); using software NLMS.")
        return _make_nlms(cfg)
    return NullEchoCanceller()


def _make_nlms(cfg: Any) -> NlmsEchoCanceller:
    return NlmsEchoCanceller(
        taps=int(getattr(cfg, "aec_nlms_taps", 256)),
        mu=float(getattr(cfg, "aec_nlms_mu", 0.3)),
    )
