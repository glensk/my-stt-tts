"""Wake-word detection (openWakeWord; custom phrase "maziko"). Phase 4.

openWakeWord is lazy-imported from the ``wake`` extra and forced onto the ONNX
backend (no tflite wheel on Apple Silicon). Until a custom "maziko" model is
trained, Phases 1-3 run on push-to-talk, so this is not yet on the critical path.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.wake")


class WakeWord:
    """Detect a single custom wake word in a stream of audio frames."""

    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from openwakeword.model import Model

            self._model = Model(wakeword_models=[self.model_path], inference_framework="onnx")

    def detect(self, frame: np.ndarray) -> bool:
        """Return ``True`` if the wake word fired on this frame."""
        self._ensure()
        scores = self._model.predict(np.asarray(frame, dtype=np.float32))
        return any(score >= self.threshold for score in scores.values())
