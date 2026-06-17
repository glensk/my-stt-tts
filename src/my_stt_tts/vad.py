"""Voice-activity detection and end-of-turn endpointing (Phase 4).

The silence-timeout state machine is pure and unit-tested. Two-stage gating
(cheap WebRTC -> Silero confirm) and Silero itself are lazy-imported from the
``vad`` extra. ``parakeet smart-turn`` model-based endpointing is layered on top
of this in Phase 4 (see PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SilenceEndpointer:
    """End the turn after ``silence_seconds`` of contiguous non-speech.

    Frame-driven and pure: feed each frame's speech flag to :meth:`update`,
    which returns ``True`` once the utterance is over. Only arms after speech has
    actually started, so leading silence never ends a turn.
    """

    silence_seconds: float
    frame_seconds: float
    _speech_started: bool = False
    _silence: float = 0.0

    def update(self, is_speech: bool) -> bool:
        """Feed one frame's speech flag; return ``True`` when the turn ends."""
        if is_speech:
            self._speech_started = True
            self._silence = 0.0
            return False
        if self._speech_started:
            self._silence += self.frame_seconds
            return self._silence >= self.silence_seconds
        return False

    def reset(self) -> None:
        self._speech_started = False
        self._silence = 0.0


class SileroVad:
    """Lazy wrapper around Silero VAD for per-frame speech detection."""

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()

    def is_speech(self, frame: Any) -> bool:
        """Return ``True`` if a frame (16 kHz chunk) contains speech."""
        import torch

        self._ensure()
        tensor = torch.as_tensor(frame, dtype=torch.float32)
        prob = float(self._model(tensor, self.sample_rate).item())
        return prob >= self.threshold
