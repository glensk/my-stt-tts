"""Wake-word detection (openWakeWord; custom phrase "maziko"). Phase 4.

openWakeWord is lazy-imported from the ``wake`` extra and forced onto the ONNX
backend (no tflite wheel on Apple Silicon). Train a "maziko" model first — see
``wakewords/WAKEWORD.md`` — then it's used by ``my-stt-tts --wake``.

The openWakeWord ``Model`` constructor changed shape across releases: modern
builds take ``wakeword_models=[...]`` plus ``inference_framework="onnx"``, while
``openwakeword==0.4.0`` (the version pinned here for arm64) takes
``wakeword_model_paths=[...]`` and has **no** ``inference_framework`` argument —
unknown kwargs fall through ``**kwargs`` into ``AudioFeatures`` and raise
``TypeError``. :class:`WakeWord` constructs the model **version-tolerantly**: it
tries the modern signature first and falls back to the 0.4.0 one on ``TypeError``
(0.4.0 infers the ONNX backend from the ``.onnx`` extension). On any
unrecoverable construction/predict failure it raises :class:`WakeUnavailable`
once so the caller can log a clear hint and stop the loop instead of spinning the
same error forever.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config

log = logging.getLogger("my_stt_tts.wake")


class WakeUnavailable(RuntimeError):
    """The wake-word model could not be loaded or run.

    Raised once (not per-frame) so a wake loop can catch it, log a single clear
    hint, and stop — rather than re-raising the same error on every audio frame.
    """


class WakeWord:
    """Detect a single custom wake word in a stream of audio frames."""

    def __init__(self, model_path: str, threshold: float = 0.5) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self._model: Any = None
        self._broken = False  # set once construction/predict fails unrecoverably
        # Max wake score from the most recent ``detect`` — surfaced by the debug
        # instrument so the log shows the per-frame score vs the threshold (so a
        # never-firing wake word is diagnosable: too high a threshold? bad audio?).
        self.last_score: float = 0.0
        self.model_name: str = Path(model_path).stem

    @classmethod
    def from_config(cls, cfg: Config) -> WakeWord:
        return cls(cfg.wake_model_path, cfg.wake_threshold)

    def available(self) -> bool:
        """True if the trained wake-word model file exists."""
        return Path(self.model_path).is_file()

    def _build_model(self) -> Any:  # noqa: ANN401 — opaque openWakeWord Model
        """Construct an openWakeWord ``Model`` across both API generations.

        Tries the modern signature (``wakeword_models=[...]``,
        ``inference_framework="onnx"``); on ``TypeError`` (the 0.4.0 ``Model``
        rejects those kwargs — they leak into ``AudioFeatures`` and raise) falls
        back to the 0.4.0 signature (``wakeword_model_paths=[...]``; 0.4.0 infers
        the ONNX backend from the ``.onnx`` extension, so no framework kwarg).
        """
        from openwakeword.model import Model

        try:
            return Model(wakeword_models=[self.model_path], inference_framework="onnx")
        except TypeError:
            # Older API (openwakeword==0.4.0): no inference_framework, and the
            # paths argument is named differently.
            return Model(wakeword_model_paths=[self.model_path])

    def _ensure(self) -> None:
        if self._model is None:
            try:
                self._model = self._build_model()
            except Exception as exc:  # noqa: BLE001 — any backend failure is terminal here
                self._broken = True
                raise WakeUnavailable(
                    f"could not load wake model {self.model_path!r}: {exc}. "
                    "Re-train the wake word (see wakewords/WAKEWORD.md) or check the "
                    "openwakeword install."
                ) from exc

    def detect(self, frame: np.ndarray) -> bool:
        """Return ``True`` if the wake word fired on this 80 ms frame.

        ``predict()`` returns a ``{model_name: score}`` dict — on 0.4.0 the key is
        the model-file stem (e.g. ``"maziko"``); we read the *values*, so the key
        naming is irrelevant. A construction or predict failure raises
        :class:`WakeUnavailable` once (not on every frame) so the loop can stop.
        """
        if self._broken:
            raise WakeUnavailable(f"wake model {self.model_path!r} is unavailable")
        self._ensure()
        try:
            scores = self._model.predict(np.asarray(frame, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 — a per-frame predict failure is terminal
            self._broken = True
            raise WakeUnavailable(f"wake model {self.model_path!r} failed to run: {exc}") from exc
        values = list(scores.values())
        self.last_score = float(max(values)) if values else 0.0
        return self.last_score >= self.threshold

    def reset(self) -> None:
        """Clear the detector's internal state between activations."""
        if self._model is not None and hasattr(self._model, "reset"):
            self._model.reset()
