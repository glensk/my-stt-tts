"""Voice-activity detection and end-of-turn endpointing (Phase 4).

The silence-timeout state machine is pure and unit-tested. Two-stage gating
(cheap WebRTC -> Silero confirm) and Silero itself are lazy-imported from the
``vad`` extra. ``parakeet smart-turn`` model-based endpointing is layered on top
of this in Phase 4 (see PLAN.md).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.vad")

# Silero VAD accepts ONLY this exact chunk size at 16 kHz; any other size raises
# "Input audio chunk is too short" inside the TorchScript model. We reframe to it.
_SILERO_FRAME_16K = 512


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
    """Lazy wrapper around Silero VAD for per-frame speech detection.

    Robust to frame size: Silero only accepts a 512-sample chunk at 16 kHz, but the
    mic/transport callbacks deliver device-blocksize chunks (often 1024/2048/4096
    samples, and the browser/satellite send whatever their worklet buffered). A
    wrongly-sized chunk raises *inside* the TorchScript model — and an unhandled
    raise in the capture loop discards the whole utterance (the "push-to-talk
    records nothing" symptom). So :meth:`is_speech` **reframes** any input to the
    model's chunk size, scores each sub-frame, and **never raises** (a model error
    reads as "no speech" rather than crashing the turn).

    ``threshold`` defaults low (0.3) so a quiet-but-present voice (a ~10% mic level)
    is still detected — a high 0.5 was treating soft speech as silence and ending
    the turn before anything was captured.
    """

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.3) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._model: Any = None
        # Last max speech probability across the reframed sub-frames — surfaced by
        # the debug instrument so the log shows *why* a frame was/ wasn't speech.
        self.last_prob: float = 0.0

    def _ensure(self) -> None:
        if self._model is None:
            from silero_vad import load_silero_vad

            self._model = load_silero_vad()

    @property
    def _frame_samples(self) -> int:
        """The exact chunk size Silero requires for this sample rate."""
        # 512 at 16 kHz; scale proportionally for other rates (256 at 8 kHz).
        return max(1, int(round(_SILERO_FRAME_16K * self.sample_rate / 16000)))

    def _prob(self, chunk: np.ndarray) -> float:
        import torch

        self._ensure()
        tensor = torch.as_tensor(chunk, dtype=torch.float32)
        return float(self._model(tensor, self.sample_rate).item())

    def is_speech(self, frame: Any) -> bool:
        """Return ``True`` if ``frame`` (any length, 16 kHz mono) contains speech.

        Reframes to the model's required chunk size and takes the loudest sub-frame
        verdict. Defensive: any model/runtime error is logged once and treated as
        non-speech so the capture loop never dies on a malformed frame.
        """
        from .audio import reframe

        arr = np.asarray(frame, dtype=np.float32).ravel()
        if arr.size == 0:
            self.last_prob = 0.0
            return False
        try:
            probs = [self._prob(chunk) for chunk in reframe(arr, self._frame_samples)]
        except Exception:  # noqa: BLE001 — a VAD failure must not abort the turn
            log.debug("Silero VAD failed on a frame; treating as silence.", exc_info=True)
            self.last_prob = 0.0
            return False
        self.last_prob = max(probs) if probs else 0.0
        return self.last_prob >= self.threshold
