"""Wake-word detection (openWakeWord; custom phrase "maziko"). Phase 4.

openWakeWord is lazy-imported from the ``wake`` extra and forced onto the ONNX
backend (no tflite wheel on Apple Silicon). Train a "maziko" model first — see
``wakewords/WAKEWORD.md`` — then it's used by ``my-stt-tts --wake``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config

log = logging.getLogger("my_stt_tts.wake")


class WakeWord:
    """Detect a single custom wake word in a stream of audio frames."""

    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self._model: Any = None

    @classmethod
    def from_config(cls, cfg: Config) -> WakeWord:
        return cls(cfg.wake_model_path, cfg.wake_threshold)

    def available(self) -> bool:
        """True if the trained wake-word model file exists."""
        return Path(self.model_path).is_file()

    def _ensure(self) -> None:
        if self._model is None:
            from openwakeword.model import Model

            self._model = Model(wakeword_models=[self.model_path], inference_framework="onnx")

    def detect(self, frame: np.ndarray) -> bool:
        """Return ``True`` if the wake word fired on this 80 ms frame."""
        self._ensure()
        scores = self._model.predict(np.asarray(frame, dtype=np.float32))
        return any(score >= self.threshold for score in scores.values())

    def reset(self) -> None:
        """Clear the detector's internal state between activations."""
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()
