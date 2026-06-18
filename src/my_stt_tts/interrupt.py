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
