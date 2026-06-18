"""End-of-turn (EOU) analysis layered on top of VAD (Phase 4/7, G2).

A :class:`TurnAnalyzer` decides when the *user* has finished speaking. Today's
fixed silence timer (:class:`~my_stt_tts.vad.SilenceEndpointer`) is one strategy;
prosodic / semantic models are another. The protocol lets the capture loop stay
agnostic:

    analyzer.reset()
    for frame in mic:
        if analyzer.update(frame, vad.is_speech(frame)):
            break  # end of turn

Strategies:

* :class:`SilenceTurnAnalyzer` — wraps the always-available silence endpointer.
* :class:`SmartTurnAnalyzer` — loads pipecat's **Smart Turn v3** ONNX model
  (``pipecat-ai/smart-turn-v3``) and, on a silence candidate, asks the model
  whether the intonation says the turn is complete. If the model file or its
  runtime deps (``onnxruntime`` + ``transformers`` WhisperFeatureExtractor) are
  missing, it **falls back** to the silence endpointer so the loop never breaks.

Selection is via :func:`make_turn_analyzer` (config-driven).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from .vad import SilenceEndpointer

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger("my_stt_tts.turn")

# Smart Turn v3 operates on the last 8 s of 16 kHz mono audio.
_SMART_TURN_SR = 16000
_SMART_TURN_WINDOW_S = 8


@runtime_checkable
class TurnAnalyzer(Protocol):
    """Decides end-of-turn from a stream of mic frames + per-frame VAD flags."""

    def update(self, frame: Any, is_speech: bool) -> bool:
        """Feed one frame and its VAD verdict; return ``True`` at end-of-turn."""
        ...

    def reset(self) -> None:
        """Reset all state for a new turn."""
        ...


class SilenceTurnAnalyzer:
    """Always-available fallback: end the turn after a silence timeout.

    Thin adapter over :class:`~my_stt_tts.vad.SilenceEndpointer` so it satisfies
    the :class:`TurnAnalyzer` protocol (the endpointer ignores the audio frame).
    """

    def __init__(self, silence_seconds: float, frame_seconds: float) -> None:
        self._endpointer = SilenceEndpointer(silence_seconds, frame_seconds=frame_seconds)

    def update(self, frame: Any, is_speech: bool) -> bool:  # noqa: ARG002 — frame unused
        return self._endpointer.update(is_speech)

    def reset(self) -> None:
        self._endpointer.reset()


class SmartTurnAnalyzer:
    """Prosodic end-of-turn detection via Smart Turn v3, silence-gated.

    The model is only consulted once the user has paused (a *silence candidate*),
    so we don't run inference every frame. On a candidate, the model scores the
    recent audio: a high "completion" probability ends the turn; a low one means
    "they're mid-thought" and we keep listening (until a longer hard-silence
    timeout forces the end). Falls back to pure silence when the model can't load.
    """

    def __init__(
        self,
        model_path: str,
        *,
        silence_seconds: float,
        frame_seconds: float,
        sample_rate: int = _SMART_TURN_SR,
        threshold: float = 0.5,
        hard_silence_seconds: float = 2.5,
    ) -> None:
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.frame_seconds = frame_seconds
        # Short silence => "candidate, ask the model"; long silence => force end.
        self._candidate = SilenceEndpointer(silence_seconds, frame_seconds=frame_seconds)
        self._hard = SilenceEndpointer(hard_silence_seconds, frame_seconds=frame_seconds)
        self._audio: list[np.ndarray] = []
        self._session: Any = None
        self._extractor: Any = None
        self._fallback = False

    # --- model loading (lazy, with graceful fallback) ---

    def _ensure_model(self) -> bool:
        """Load the ONNX session + feature extractor once. Returns availability."""
        if self._fallback:
            return False
        if self._session is not None:
            return True
        if not Path(self.model_path).is_file():
            log.info(
                "Smart Turn model not found at %s; falling back to silence endpointing.",
                self.model_path,
            )
            self._fallback = True
            return False
        try:
            import onnxruntime  # noqa: PLC0415 — heavy, lazy
            from transformers import WhisperFeatureExtractor  # noqa: PLC0415

            opts = onnxruntime.SessionOptions()
            opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = onnxruntime.InferenceSession(self.model_path, sess_options=opts)
            self._extractor = WhisperFeatureExtractor(
                feature_size=80, sampling_rate=self.sample_rate, chunk_length=_SMART_TURN_WINDOW_S
            )
            return True
        except Exception:  # missing runtime / load failure must not break the loop
            log.warning("Smart Turn unavailable; falling back to silence.", exc_info=True)
            self._fallback = True
            return False

    # --- inference ---

    def _completion_probability(self) -> float:
        """Run the model on the last 8 s of buffered audio -> P(turn complete)."""
        audio = np.concatenate(self._audio) if self._audio else np.zeros(0, dtype=np.float32)
        max_samples = _SMART_TURN_WINDOW_S * self.sample_rate
        if audio.shape[0] > max_samples:
            audio = audio[-max_samples:]
        features = self._extractor(
            audio,
            sampling_rate=self.sample_rate,
            padding="max_length",
            max_length=max_samples,
            return_tensors="np",
            do_normalize=True,
        )
        feats = np.asarray(features["input_features"], dtype=np.float32)
        outputs = self._session.run(None, {"input_features": feats})
        return float(np.asarray(outputs[0]).ravel()[0])

    # --- TurnAnalyzer protocol ---

    def update(self, frame: Any, is_speech: bool) -> bool:
        arr = np.asarray(frame, dtype=np.float32).ravel()
        if arr.size:
            self._audio.append(arr)
        # Hard timeout always wins (and is the fallback path's only signal).
        if self._hard.update(is_speech):
            return True
        candidate = self._candidate.update(is_speech)
        if not candidate:
            return False
        # We have a short pause; if the model is available, let prosody decide.
        if not self._ensure_model():
            return True  # fallback == plain silence endpointing
        try:
            prob = self._completion_probability()
        except Exception:  # inference hiccup -> behave like the silence endpointer
            log.debug("Smart Turn inference failed; ending on silence.", exc_info=True)
            return True
        if prob >= self.threshold:
            return True
        # Model says "not done": re-arm the short candidate timer and keep going.
        self._candidate.reset()
        return False

    def reset(self) -> None:
        self._candidate.reset()
        self._hard.reset()
        self._audio = []


def make_turn_analyzer(cfg: Config, frame_seconds: float) -> TurnAnalyzer:
    """Build the configured :class:`TurnAnalyzer` (``cfg.turn_analyzer``).

    ``"smart"`` selects :class:`SmartTurnAnalyzer` (which itself falls back to
    silence if the model is absent); anything else selects the pure silence timer.
    """
    if cfg.turn_analyzer == "smart":
        return SmartTurnAnalyzer(
            cfg.smart_turn_model_path,
            silence_seconds=cfg.vad_silence_seconds,
            frame_seconds=frame_seconds,
            sample_rate=cfg.sample_rate,
            threshold=cfg.smart_turn_threshold,
        )
    return SilenceTurnAnalyzer(cfg.vad_silence_seconds, frame_seconds=frame_seconds)
