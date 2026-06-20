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

Phase diversity (the "fires offline, never live" fix)
------------------------------------------------------
openWakeWord scores once per 1280-sample (80 ms) frame, locked to ONE phase
relative to the spoken word. The maziko score swings ~25x (≈0.03..0.85) purely
with where that frame boundary falls. In an always-listening loop the frame grid
is fixed by capture timing, so a single utterance gets exactly ONE phase: an
unlucky alignment scores ~0.03 and never fires even though the SAME audio at a
better offset scores ~0.7. :class:`WakeWord` therefore runs ``phases`` detectors
fed the same audio but each offset by ``1280 / phases`` samples, and fires on the
MAX score over all of them — covering the phase space. Measured to lift recall
from 2/8 to 5/8 synthesized voices with no extra false-positives and a 0.22
real-time factor at 8 phases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config

log = logging.getLogger("my_stt_tts.wake")

# openWakeWord's fixed frame size: 1280 samples == 80 ms at 16 kHz. Every phase
# detector consumes exact multiples of this; phase offsets are sub-multiples of it.
FRAME_SAMPLES = 1280


def to_int16_pcm(frame: np.ndarray) -> np.ndarray:
    """Convert audio to int16 PCM (±32768) for openWakeWord, scaling float input.

    openWakeWord 0.4.0's ``AudioFeatures`` requires 16-bit-int samples: it buffers
    the raw audio as a Python list and re-casts it with ``np.array(...).astype(
    np.int16)``, so a **float32 signal in [-1, 1] is truncated to all zeros** and the
    model sees silence (score pinned at ≈0.001 — the never-fires bug). The rest of
    this pipeline carries float32 mono, so the float→int16 scale conversion happens
    here at the model boundary. A frame that is already ``int16`` is passed through
    unchanged; a float frame is clipped to [-1, 1] and scaled by 32767 (the same
    convention as openWakeWord's own ``detect_from_microphone.py`` example).
    """
    arr = np.asarray(frame)
    if arr.dtype == np.int16:
        return arr
    return (np.clip(arr.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)


class WakeUnavailable(RuntimeError):
    """The wake-word model could not be loaded or run.

    Raised once (not per-frame) so a wake loop can catch it, log a single clear
    hint, and stop — rather than re-raising the same error on every audio frame.
    """


class WakeWord:
    """Detect a single custom wake word in a stream of audio frames.

    Holds ``phases`` openWakeWord models fed the same audio at staggered sub-frame
    offsets so the wake word is scored at every frame phase (the recall fix — see
    the module docstring). With ``phases == 1`` this is exactly the classic
    single-detector behaviour.
    """

    def __init__(self, model_path: str, threshold: float = 0.5, *, phases: int = 1) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.phases = max(1, int(phases))
        self._models: list[Any] = []
        self._broken = False  # set once construction/predict fails unrecoverably
        # Per-phase rolling buffers of int16 samples not yet formed into a 1280 frame.
        # The i-th detector starts at sample offset i*(1280/phases): we prime its
        # buffer with that many leading zeros so its frame grid is shifted.
        self._pending: list[np.ndarray] = []
        # Max wake score from the most recent ``detect`` — surfaced by the debug
        # instrument so the log shows the per-frame score vs the threshold (so a
        # never-firing wake word is diagnosable: too high a threshold? bad audio?).
        self.last_score: float = 0.0
        self.model_name: str = Path(model_path).stem

    @classmethod
    def from_config(cls, cfg: Config) -> WakeWord:
        return cls(cfg.wake_model_path, cfg.wake_threshold, phases=cfg.wake_phases)

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
        if not self._models:
            try:
                self._models = [self._build_model() for _ in range(self.phases)]
            except Exception as exc:  # noqa: BLE001 — any backend failure is terminal here
                self._broken = True
                self._models = []
                raise WakeUnavailable(
                    f"could not load wake model {self.model_path!r}: {exc}. "
                    "Re-train the wake word (see wakewords/WAKEWORD.md) or check the "
                    "openwakeword install."
                ) from exc
            self._reset_pending()

    def _reset_pending(self) -> None:
        """Prime each phase detector's buffer with its leading-offset zeros.

        Detector ``i`` is shifted by ``i * (1280 / phases)`` samples so the K grids
        collectively cover the 1280-sample phase space. The lead zeros simply move
        where the first full frame boundary lands for that detector.
        """
        hop = FRAME_SAMPLES // self.phases
        self._pending = [np.zeros(i * hop, dtype=np.int16) for i in range(self.phases)]

    def detect(self, frame: np.ndarray) -> bool:
        """Return ``True`` if the wake word fired on this 80 ms frame.

        ``predict()`` returns a ``{model_name: score}`` dict — on 0.4.0 the key is
        the model-file stem (e.g. ``"maziko"``); we read the *values*, so the key
        naming is irrelevant. A construction or predict failure raises
        :class:`WakeUnavailable` once (not on every frame) so the loop can stop.

        The frame is converted to **int16 PCM** before scoring (see
        :func:`to_int16_pcm`). openWakeWord 0.4.0's ``AudioFeatures`` *requires*
        16-bit-int input — its melspectrogram path buffers the raw samples as a
        Python list and re-casts them with ``np.array(...).astype(np.int16)``, which
        silently **truncates a float32 [-1, 1] signal to all zeros** (so the model
        sees near-silence and the score is pinned at ≈0.001 — the never-fires bug).
        The rest of the pipeline (capture/VAD/STT) is float32, so the conversion is
        done here, at the model boundary, and nowhere else.

        With ``phases > 1`` the incoming samples are fanned out to every staggered
        detector; ``last_score`` is the MAX over all detectors that produced a fresh
        frame this call, so a phase-unlucky utterance still fires (the live-recall
        fix — see the module docstring).
        """
        if self._broken:
            raise WakeUnavailable(f"wake model {self.model_path!r} is unavailable")
        self._ensure()
        pcm = to_int16_pcm(frame)
        best = 0.0
        scored = False
        for i, model in enumerate(self._models):
            self._pending[i] = np.concatenate([self._pending[i], pcm])
            while self._pending[i].size >= FRAME_SAMPLES:
                chunk = self._pending[i][:FRAME_SAMPLES]
                self._pending[i] = self._pending[i][FRAME_SAMPLES:]
                try:
                    scores = model.predict(chunk)
                except Exception as exc:  # noqa: BLE001 — a per-frame predict failure is terminal
                    self._broken = True
                    raise WakeUnavailable(
                        f"wake model {self.model_path!r} failed to run: {exc}"
                    ) from exc
                values = list(scores.values())
                if values:
                    best = max(best, float(max(values)))
                    scored = True
        if scored:
            self.last_score = best
        return self.last_score >= self.threshold

    def reset(self) -> None:
        """Clear the detector's internal state between activations.

        Resets openWakeWord's prediction buffer on every phase model AND re-primes
        the per-phase staggered input buffers, so a fresh listen session starts
        clean (no stale frame straddling the phase boundaries from the last one).
        """
        for model in self._models:
            if hasattr(model, "reset"):
                model.reset()
        if self._models:
            self._reset_pending()
        self.last_score = 0.0
