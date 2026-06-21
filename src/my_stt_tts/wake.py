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
from typing import Any, Literal, overload

import numpy as np

from .config import WAKEWORDS_DIR, Config, wake_model_for

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


class OrCombinedWake:
    """openWakeWord OR sherpa-KWS — fire if EITHER detector fires (custom words only).

    Presents the EXACT :class:`WakeWord` surface the wake loop drives —
    ``detect(frame) -> bool``, ``reset()``, ``last_score``, ``threshold``,
    ``model_name``, ``available()`` — so :func:`my_stt_tts.audio.listen_for_wake` calls
    it identically. It wraps the openWakeWord :class:`WakeWord` (always) plus an optional
    :class:`my_stt_tts.kws.SherpaKws` (only for a CUSTOM / self-trained word, when KWS is
    enabled + available). Each frame is fed to BOTH; the word fires the instant EITHER
    fires. ``last_detector`` ("oww" | "kws") names which produced the latest fire so the
    caller can report it. For an OFFICIAL word this class is NEVER constructed (the loop
    uses the bare :class:`WakeWord`), so official behaviour is byte-identical.
    """

    def __init__(self, oww: WakeWord, kws: Any = None) -> None:  # noqa: ANN401 — SherpaKws | None
        self.oww = oww
        self.kws = kws
        # last_detector: which detector produced the most recent fire ("oww" | "kws"),
        # or "" when nothing has fired this session. Surfaced on the detection event.
        self.last_detector: str = ""

    @property
    def threshold(self) -> float:
        return self.oww.threshold

    @property
    def model_name(self) -> str:
        return self.oww.model_name

    @property
    def last_score(self) -> float:
        """The oWW score (the continuous one the debug instrument plots).

        KWS has no continuous score (the transducer reports a matched-keyword label, not a
        probability), so the plotted score stays the openWakeWord one; a KWS-only fire is
        reflected via :attr:`last_detector` and :meth:`detect` returning ``True``.
        """
        return self.oww.last_score

    def available(self) -> bool:
        return self.oww.available()

    def detect(self, frame: np.ndarray) -> bool:
        """Fire if EITHER detector fires on this frame. oWW is scored first (cheap, keeps
        ``last_score`` live for the debug plot); KWS only when present. ``last_detector``
        records the winner — oWW wins ties (it ran first)."""
        oww_fired = self.oww.detect(frame)
        if oww_fired:
            self.last_detector = "oww"
            return True
        if self.kws is not None and self.kws.detect(frame):
            self.last_detector = "kws"
            return True
        return False

    def reset(self) -> None:
        self.oww.reset()
        if self.kws is not None:
            self.kws.reset()
        self.last_detector = ""


def make_wake_detector(cfg: Config) -> WakeWord | OrCombinedWake:
    """Build the wake detector for ``cfg.wake_phrase``: bare oWW, or oWW-OR-KWS.

    The single entry point the wake loop uses. For an OFFICIAL word it returns the bare
    :class:`WakeWord` (openWakeWord only — byte-identical to before). For a CUSTOM /
    self-trained word, when ``kws_enabled`` and the sherpa KeywordSpotter is available, it
    returns an :class:`OrCombinedWake` that fires if EITHER detector fires; otherwise it
    also returns the bare :class:`WakeWord`. KWS construction is fully defensive (returns
    ``None`` on any failure), so this NEVER raises and never changes oWW behaviour.
    """
    oww = WakeWord.from_config(cfg)
    from .config import is_official_wake_word

    if is_official_wake_word(cfg.wake_phrase) or not getattr(cfg, "kws_enabled", True):
        return oww
    from .kws import SherpaKws

    kws = SherpaKws.from_config(cfg, cfg.wake_phrase)
    if kws is None:
        return oww
    log.info("wake detector: openWakeWord OR sherpa-KWS for custom word %r", cfg.wake_phrase)
    return OrCombinedWake(oww, kws)


@overload
def score_wake_clip(
    clip: np.ndarray,
    sample_rate: int,
    word: str,
    *,
    threshold: float = ...,
    phases: int = ...,
    gain: float = ...,
    wakewords_dir: str = ...,
    with_trace: Literal[False] = ...,
) -> tuple[float, bool]: ...


@overload
def score_wake_clip(
    clip: np.ndarray,
    sample_rate: int,
    word: str,
    *,
    threshold: float = ...,
    phases: int = ...,
    gain: float = ...,
    wakewords_dir: str = ...,
    with_trace: Literal[True],
) -> tuple[float, bool, list[float]]: ...


def score_wake_clip(
    clip: np.ndarray,
    sample_rate: int,
    word: str,
    *,
    threshold: float = 0.4,
    phases: int = 8,
    gain: float = 1.0,
    wakewords_dir: str = WAKEWORDS_DIR,
    with_trace: bool = False,
) -> tuple[float, bool] | tuple[float, bool, list[float]]:
    """Score a recorded clip against the wake model for ``word`` (the GUI diagnostic).

    Loads the :class:`WakeWord` for ``word`` via :func:`wake_model_for` — NOT
    necessarily the configured wake word — resamples ``clip`` to the 16 kHz the
    model expects, applies ``gain`` (clip-protected to ±1.0 — the gain-sweep knob
    that proves a too-quiet capture is the cause), reframes it to 1280-sample (80 ms)
    frames, and feeds them through the REAL :meth:`WakeWord.detect` path
    frame-by-frame (so it is phase-diverse, exactly what the always-listening loop
    sees). Returns ``(confidence, fired)`` where ``confidence`` is the MAX
    ``last_score`` over the whole clip and ``fired`` is ``confidence >= threshold``.

    ``with_trace=True`` returns ``(confidence, fired, score_trace)`` instead, where
    ``score_trace`` is the per-frame MAX-over-phases score across the clip — the
    trace the GUI draws under the waveform so a localized spike (the word IS scoring,
    just sub-threshold) is distinguishable from a flat-zero dead capture.

    Defensive: an empty clip, a missing model file, or an unavailable openWakeWord
    backend all return zero confidence (and an empty trace) rather than raising — the
    caller turns that into a clear "model unavailable" message instead of crashing.
    """
    from .audio import apply_gain, reframe, resample_to

    def _result(conf: float, fired: bool, trace: list[float]) -> Any:  # noqa: ANN401
        return (conf, fired, trace) if with_trace else (conf, fired)

    model_path = wake_model_for(word, wakewords_dir)
    if not Path(model_path).is_file():
        return _result(0.0, False, [])
    arr = np.asarray(clip, dtype=np.float32).ravel()
    if arr.size == 0:
        return _result(0.0, False, [])
    arr = resample_to(arr, int(sample_rate), 16000)
    if gain != 1.0:
        arr = apply_gain(arr, gain)
    detector = WakeWord(model_path, threshold, phases=phases)
    best = 0.0
    trace: list[float] = []
    try:
        for frame in reframe(arr, FRAME_SAMPLES):
            detector.detect(frame)
            trace.append(round(detector.last_score, 4))
            best = max(best, detector.last_score)
    except WakeUnavailable as exc:
        log.warning("wake-test scoring unavailable for %r: %s", word, exc)
        return _result(0.0, False, [])
    return _result(best, best >= threshold, trace)


def _kws_fires_on_clip(clip: np.ndarray, sample_rate: int, word: str, cfg: Config) -> bool:
    """Stream ``clip`` through the sherpa KWS for ``word`` and return whether it fired.

    Builds the (defensive) :class:`my_stt_tts.kws.SherpaKws` for ``word`` from ``cfg``,
    resamples to 16 kHz, feeds it in :data:`FRAME_SAMPLES` chunks, then a trailing-silence
    flush so the last word decodes. Returns ``False`` when KWS is unavailable / the clip is
    empty — never raises. The caller has already established ``word`` is custom + enabled.
    """
    from .audio import resample_to
    from .kws import SherpaKws

    kws = SherpaKws.from_config(cfg, word)
    arr = np.asarray(clip, dtype=np.float32).ravel()
    if kws is None or arr.size == 0:
        return False
    arr = resample_to(arr, int(sample_rate), 16000)
    kws.reset()
    fired = any(
        kws.detect(arr[start : start + FRAME_SAMPLES])
        for start in range(0, arr.size, FRAME_SAMPLES)
    )
    return fired or kws.flush()


def score_wake_clip_combined(
    clip: np.ndarray,
    sample_rate: int,
    word: str,
    cfg: Config,
    *,
    wakewords_dir: str = WAKEWORDS_DIR,
) -> tuple[float, bool, str, list[float]]:
    """OR-combined clip scoring: ``(confidence, fired, detector, score_trace)``.

    Runs the openWakeWord :func:`score_wake_clip` (phase-diverse, the same as the live
    loop) AND — for a CUSTOM / self-trained word, when ``cfg.kws_enabled`` and the sherpa
    KeywordSpotter is available — ALSO scores the clip through KWS, firing if EITHER fires.
    ``detector`` names which produced the fire: ``"oww"`` (oWW fired — it wins even if KWS
    also would, as it is the primary continuous-score path), ``"kws"`` (KWS-only fire), or
    ``""`` when neither fired. ``confidence`` / ``score_trace`` stay the openWakeWord values
    (KWS has no continuous score), so the GUI's level meter + trace are unchanged.

    For an OFFICIAL word this is byte-identical to :func:`score_wake_clip` with the detector
    forced to ``"oww"`` — KWS is NEVER consulted (the guardrail). Fully defensive: a KWS
    failure simply leaves the oWW result intact.
    """
    from .config import is_official_wake_word

    conf, oww_fired, trace = score_wake_clip(
        clip,
        sample_rate,
        word,
        threshold=cfg.wake_threshold,
        phases=cfg.wake_phases,
        wakewords_dir=wakewords_dir,
        with_trace=True,
    )
    if oww_fired:
        return (conf, True, "oww", trace)
    if is_official_wake_word(word) or not getattr(cfg, "kws_enabled", True):
        return (conf, False, "", trace)
    # Custom word, oWW did NOT fire: give the open-vocabulary KWS a shot.
    if _kws_fires_on_clip(clip, sample_rate, word, cfg):
        return (conf, True, "kws", trace)
    return (conf, False, "", trace)
