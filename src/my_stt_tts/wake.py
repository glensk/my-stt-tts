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

Temporal smoothing (microWakeWord's runtime idea, ported into the LIVE path)
----------------------------------------------------------------------------
The MAX-over-phases gives ONE score per 80 ms frame; the live detector used to
fire the instant a SINGLE frame's score cleared the threshold. microWakeWord
instead keeps a sliding window of the recent per-frame probabilities and fires on
their MOVING AVERAGE (its ``process_streaming_prob``), which tolerates a one-frame
dip and suppresses a one-frame spike. :class:`WakeWord` ports that: it keeps a
``collections.deque(maxlen=wake_window)`` of the per-frame MAX-over-phases score and
fires when ``mean(window) >= threshold``. After a fire it observes a
``wake_refractory``-frame lockout (the same refractory :func:`count_fires` applies
offline) so one utterance can't re-fire mid-word. ``wake_window == 1`` (and
``wake_refractory == 0``) is **byte-identical** to the old single-frame behaviour —
``mean([s]) == s``, so the default keeps every working model firing exactly as
before — and the OFFLINE eval (:func:`score_wake_clip` / :func:`fa_eval`) uses the
SAME moving-average criterion via :func:`moving_average_fires`, so the FA/hour ROC-DET
numbers are TRUE OF the live detector (live == eval). Averaging RAISES the firing
bar, so a window > 1 ships as the default ONLY where it is empirically shown not to
regress recall — see ``PLAN_wake_checker_loop.md``.
"""

from __future__ import annotations

import collections
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

    Fires on the MOVING AVERAGE of the last ``window`` per-frame MAX-over-phases
    scores (microWakeWord's runtime smoothing) and observes a ``refractory``-frame
    lockout after each fire. ``window == 1`` + ``refractory == 0`` (the defaults) is
    byte-identical to firing on the single most recent frame, so the default path is
    unchanged. See the module docstring.
    """

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.5,
        *,
        phases: int = 1,
        window: int = 1,
        refractory: int = 0,
        custom_verifier: Any = None,  # noqa: ANN401 — my_stt_tts.wake_verifier.CustomVerifier
        verifier_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.threshold = threshold
        self.phases = max(1, int(phases))
        # Sliding-window moving-average fire criterion (microWakeWord's
        # process_streaming_prob): fire when mean(last `window` per-frame scores) >=
        # threshold. window == 1 collapses to "fire on the single most recent frame"
        # (mean([s]) == s) — byte-identical to the old behaviour.
        self.window = max(1, int(window))
        # Refractory lockout: suppress re-fires for `refractory` frames after a fire
        # (reuses count_fires' refractory idea so one utterance can't re-fire mid-word).
        # 0 == no lockout (the old behaviour).
        self.refractory = max(0, int(refractory))
        # The moving-average ring buffer of per-frame MAX-over-phases scores, and the
        # remaining lockout frames. Both cleared by reset().
        self._score_window: collections.deque[float] = collections.deque(maxlen=self.window)
        self._refractory_left = 0
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
        # Optional custom verifier (Task 3): when set, a base-model fire is GATED — it
        # only counts when the verifier ALSO confirms the recent audio is the enrolled
        # word (verifier_prob >= verifier_threshold). None => the base model fires
        # alone, byte-identical to before. The verifier scores a rolling window of the
        # most recent ~1.5 s of float audio buffered here.
        self.custom_verifier = custom_verifier
        self.verifier_threshold = float(verifier_threshold)
        self.last_verifier_score: float = 0.0
        self._verify_window = np.zeros(0, dtype=np.float32)

    @classmethod
    def from_config(cls, cfg: Config) -> WakeWord:
        """Build the detector from ``cfg``, auto-loading a trained custom verifier if present.

        When a verifier has been trained for ``cfg.wake_phrase`` (see
        :mod:`my_stt_tts.wake_verifier`), it is loaded and gates the base prediction.
        Loading is fully defensive — a missing file / missing scikit-learn yields
        ``None`` and the detector runs ungated, byte-identical to before.
        """
        verifier = None
        try:
            from .wake_verifier import CustomVerifier

            verifier = CustomVerifier.load(cfg.wake_phrase)
            if verifier is not None:
                log.info("wake: loaded custom verifier for %r", cfg.wake_phrase)
        except Exception as exc:  # noqa: BLE001 — verifier is best-effort; never block the loop
            log.debug("custom verifier not loaded for %r: %s", cfg.wake_phrase, exc)
        return cls(
            cfg.wake_model_path,
            cfg.wake_threshold,
            phases=cfg.wake_phases,
            window=getattr(cfg, "wake_window", 1),
            refractory=getattr(cfg, "wake_refractory", 0),
            custom_verifier=verifier,
        )

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

        The fire decision is the MOVING AVERAGE of the last :attr:`window` per-frame
        MAX-over-phases scores: it fires when ``mean(window) >= threshold`` and the
        detector is not in its post-fire :attr:`refractory` lockout (microWakeWord's
        runtime smoothing — see the module docstring). ``window == 1`` makes
        ``mean([last_score]) == last_score``, so the criterion is identical to the old
        single-frame ``last_score >= threshold`` and ``refractory == 0`` adds no
        lockout — the default is byte-identical.

        When a :attr:`custom_verifier` is attached (Task 3), a (windowed) fire is GATED:
        the most recent ~1.5 s of audio (buffered here) is scored by the verifier and
        the wake only fires when ``verifier_prob >= verifier_threshold`` too, so a
        false base trigger that doesn't match the enrolled voice is rejected. With no
        verifier this branch is skipped and behaviour is byte-identical.
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
            # Feed the moving-average ring buffer ONLY on a fresh frame, so an
            # unscored warmup call (no full 1280-frame yet) does not re-push a stale
            # value and inflate the mean.
            self._score_window.append(best)
        if self.custom_verifier is not None:
            self._buffer_for_verifier(frame)
        # During the post-fire refractory lockout, never fire (consume one frame). The
        # window keeps filling above so the moving average stays current.
        if self._refractory_left > 0:
            self._refractory_left -= 1
            return False
        windowed_fired = self._window_mean() >= self.threshold
        if not windowed_fired:
            return False
        if self.custom_verifier is not None and not self._verifier_confirms():
            return False
        self._refractory_left = self.refractory
        return True

    def _window_mean(self) -> float:
        """Mean of the moving-average window (== ``last_score`` while window holds 1).

        Empty (no frame scored yet) reads as ``0.0`` so the detector cannot fire before
        it has seen any audio. With ``window == 1`` this is exactly ``last_score``.
        """
        if not self._score_window:
            return 0.0
        return float(sum(self._score_window) / len(self._score_window))

    # The verifier scores a rolling window of the most recent audio; ~1.5 s is enough
    # to contain a wake word while staying cheap to embed.
    _VERIFY_WINDOW_SAMPLES = int(1.5 * 16000)

    def _buffer_for_verifier(self, frame: np.ndarray) -> None:
        """Append ``frame`` (float, 16 kHz) to the rolling verifier window."""
        arr = np.asarray(frame, dtype=np.float32).ravel()
        self._verify_window = np.concatenate([self._verify_window, arr])[
            -self._VERIFY_WINDOW_SAMPLES :
        ]

    def _verifier_confirms(self) -> bool:
        """Whether the custom verifier confirms the buffered window is the enrolled word."""
        self.last_verifier_score = float(self.custom_verifier.score(self._verify_window, 16000))
        return self.last_verifier_score >= self.verifier_threshold

    def reset(self) -> None:
        """Clear the detector's internal state between activations.

        Resets openWakeWord's prediction buffer on every phase model AND re-primes
        the per-phase staggered input buffers, so a fresh listen session starts
        clean (no stale frame straddling the phase boundaries from the last one).
        Also clears the moving-average window and the refractory lockout so the next
        session's first frames are scored from scratch.
        """
        for model in self._models:
            if hasattr(model, "reset"):
                model.reset()
        if self._models:
            self._reset_pending()
        self.last_score = 0.0
        self.last_verifier_score = 0.0
        self._verify_window = np.zeros(0, dtype=np.float32)
        self._score_window.clear()
        self._refractory_left = 0


class OrCombinedWake:
    """openWakeWord OR sherpa-KWS OR few-shot enrolled — fire if ANY fires (custom words).

    Presents the EXACT :class:`WakeWord` surface the wake loop drives —
    ``detect(frame) -> bool``, ``reset()``, ``last_score``, ``threshold``,
    ``model_name``, ``available()`` — so :func:`my_stt_tts.audio.listen_for_wake` calls
    it identically. It wraps the openWakeWord :class:`WakeWord` (always) plus, for a CUSTOM /
    self-trained word, an optional :class:`my_stt_tts.kws.SherpaKws` (zero-train open-vocab)
    AND an optional :class:`my_stt_tts.enrolled_wake.EnrolledWake` (few-shot on the user's own
    clips). Each frame is fed to all present detectors; the word fires the instant ANY fires.
    ``last_detector`` ("oww" | "kws" | "fewshot") names which produced the latest fire so the
    caller can report it. For an OFFICIAL word this class is NEVER constructed (the loop uses
    the bare :class:`WakeWord`), so official behaviour is byte-identical.
    """

    def __init__(
        self,
        oww: WakeWord,
        kws: Any = None,  # noqa: ANN401 — SherpaKws | None
        fewshot: Any = None,  # noqa: ANN401 — EnrolledWake | None
    ) -> None:
        self.oww = oww
        self.kws = kws
        self.fewshot = fewshot
        # last_detector: which detector produced the most recent fire ("oww" | "kws" |
        # "fewshot"), or "" when nothing has fired this session. Surfaced on the event.
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
        """Fire if ANY detector fires on this frame. oWW is scored first (cheap, keeps
        ``last_score`` live for the debug plot); KWS then the few-shot enrolled detector
        only when present. ALL present detectors are fed the frame every call (so each keeps
        its rolling state current) before the decision. ``last_detector`` records the winner —
        oWW wins ties (it ran first), then KWS, then few-shot."""
        oww_fired = self.oww.detect(frame)
        kws_fired = self.kws.detect(frame) if self.kws is not None else False
        fewshot_fired = self.fewshot.detect(frame) if self.fewshot is not None else False
        if oww_fired:
            self.last_detector = "oww"
            return True
        if kws_fired:
            self.last_detector = "kws"
            return True
        if fewshot_fired:
            self.last_detector = "fewshot"
            return True
        return False

    def reset(self) -> None:
        self.oww.reset()
        if self.kws is not None:
            self.kws.reset()
        if self.fewshot is not None:
            self.fewshot.reset()
        self.last_detector = ""


def make_wake_detector(cfg: Config) -> WakeWord | OrCombinedWake:
    """Build the wake detector for ``cfg.wake_phrase``: bare oWW, or oWW OR'd with extras.

    The single entry point the wake loop uses. For an OFFICIAL word it returns the bare
    :class:`WakeWord` (openWakeWord only — byte-identical to before). For a CUSTOM /
    self-trained word it OR-combines openWakeWord with any AVAILABLE second-stage detector:
    the sherpa :class:`my_stt_tts.kws.SherpaKws` (zero-train open-vocab, when ``kws_enabled``)
    and the few-shot :class:`my_stt_tts.enrolled_wake.EnrolledWake` (when ``fewshot_wake_enabled``
    and the word has saved enrolled references). When neither extra is available it returns the
    bare :class:`WakeWord`. Every extra's construction is fully defensive (returns ``None`` on
    any failure / when not applicable), so this NEVER raises and never changes oWW behaviour.
    """
    oww = WakeWord.from_config(cfg)
    from .config import is_official_wake_word

    if is_official_wake_word(cfg.wake_phrase):
        return oww  # official words are openWakeWord-ONLY (byte-identical)
    kws = None
    if getattr(cfg, "kws_enabled", True):
        from .kws import SherpaKws

        kws = SherpaKws.from_config(cfg, cfg.wake_phrase)
    fewshot = None
    if getattr(cfg, "fewshot_wake_enabled", True):
        from .enrolled_wake import EnrolledWake

        fewshot = EnrolledWake.from_config(cfg, cfg.wake_phrase)
    if kws is None and fewshot is None:
        return oww
    branches = [name for name, d in (("KWS", kws), ("few-shot", fewshot)) if d is not None]
    log.info(
        "wake detector: openWakeWord OR %s for custom word %r",
        " OR ".join(branches),
        cfg.wake_phrase,
    )
    return OrCombinedWake(oww, kws, fewshot)


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
    window: int = ...,
    refractory: int = ...,
    patience: int = ...,
    debounce: int = ...,
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
    window: int = ...,
    refractory: int = ...,
    patience: int = ...,
    debounce: int = ...,
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
    window: int = 1,
    refractory: int = 0,
    patience: int = 1,
    debounce: int = 0,
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
    ``last_score`` over the whole clip.

    ``fired`` is the firing decision, computed under the SAME moving-average criterion
    the LIVE detector uses (so the offline number is TRUE of the running loop —
    ``live == eval``). With the defaults (``window <= 1``, ``refractory <= 0``,
    ``patience <= 1``, ``debounce <= 0``) it is ``confidence >= threshold`` —
    byte-identical to before. When ``window > 1`` or ``refractory > 0`` the trace is
    replayed under the live moving-average + refractory criterion
    (:func:`moving_average_fires`). The legacy ``patience`` / ``debounce`` knobs
    (consecutive-frame de-bounce, :func:`fired_with_patience`) are still honoured when
    set, for callers that drive that path; ``window`` / ``refractory`` take precedence
    as they mirror the live detector.

    ``with_trace=True`` returns ``(confidence, fired, score_trace)`` instead, where
    ``score_trace`` is the per-frame MAX-over-phases score across the clip — the
    trace the GUI draws under the waveform so a localized spike (the word IS scoring,
    just sub-threshold) is distinguishable from a flat-zero dead capture. It is also
    the input to :func:`count_fa_events`, :func:`moving_average_fires`, and
    :func:`fired_with_patience`.

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
    if window > 1 or refractory > 0:
        # The live detector's exact criterion (moving average + refractory): live == eval.
        fired = moving_average_fires(trace, threshold, window=window, refractory=refractory)
    elif patience > 1 or debounce > 0:
        fired = fired_with_patience(trace, threshold, patience=patience, debounce=debounce)
    else:
        fired = best >= threshold
    return _result(best, fired, trace)


def count_fires(
    trace: list[float] | np.ndarray,
    threshold: float,
    *,
    patience: int = 1,
    debounce: int = 0,
) -> int:
    """Count distinct FIRES in a score ``trace`` under ship-config patience/debounce.

    A fire needs ``patience`` CONSECUTIVE frames at-or-above ``threshold`` (openWakeWord's
    de-bouncing knob — it suppresses a one-frame fluke); after each fire a ``debounce``
    refractory window of frames is skipped so the live loop does not re-fire mid-utterance.
    Pure; ``patience <= 1`` + ``debounce <= 0`` collapses to "one fire per contiguous
    above-threshold run". The bool gate :func:`fired_with_patience` is ``count_fires > 0``.
    """
    arr = np.asarray(trace, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0
    pat = max(1, int(patience))
    deb = max(0, int(debounce))
    fires = 0
    run = 0
    refractory = 0
    for score in arr:
        if refractory > 0:
            refractory -= 1
            run = 0
            continue
        if score >= threshold:
            run += 1
            if run >= pat:
                fires += 1
                refractory = deb  # skip the refractory window before the next fire
                run = 0
        else:
            run = 0
    return fires


def fired_with_patience(
    trace: list[float] | np.ndarray,
    threshold: float,
    *,
    patience: int = 1,
    debounce: int = 0,
) -> bool:
    """Whether a per-frame score ``trace`` fires AT LEAST ONCE under patience/debounce.

    The replay-under-ship-config gate (Task 5): instead of "any single frame cleared the
    threshold", a fire requires ``patience`` CONSECUTIVE frames at-or-above ``threshold``
    (suppresses a one-frame fluke) and honours the ``debounce`` refractory window — i.e.
    ``count_fires(...) > 0``. ``patience <= 1`` + ``debounce <= 0`` reproduces the classic
    "any frame >= threshold" decision, so the default behaviour is unchanged. Pure.
    """
    return count_fires(trace, threshold, patience=patience, debounce=debounce) > 0


def count_fires_moving_average(
    trace: list[float] | np.ndarray,
    threshold: float,
    *,
    window: int = 1,
    refractory: int = 0,
) -> int:
    """Count distinct FIRES in a score ``trace`` under the LIVE moving-average criterion.

    Replays EXACTLY what :meth:`WakeWord.detect` does frame-by-frame: a fire when the
    MEAN of the trailing ``window`` per-frame scores is at-or-above ``threshold``, then a
    ``refractory``-frame lockout (during which scores still flow into the moving average,
    but no fire is emitted). This is the offline twin of the live detector — so an eval
    that calls it (or :func:`moving_average_fires`) measures the TRUE live behaviour
    (``live == eval``). ``window <= 1`` + ``refractory <= 0`` collapses to "one fire per
    contiguous above-threshold run" — byte-identical to the old single-frame decision and
    to :func:`count_fires` with ``patience == 1`` / ``debounce == 0``. Pure.
    """
    arr = np.asarray(trace, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0
    win = max(1, int(window))
    refr = max(0, int(refractory))
    ring: collections.deque[float] = collections.deque(maxlen=win)
    fires = 0
    refractory_left = 0
    for score in arr:
        ring.append(float(score))
        if refractory_left > 0:
            refractory_left -= 1
            continue
        if (sum(ring) / len(ring)) >= threshold:
            fires += 1
            refractory_left = refr  # lock out the refractory window before the next fire
    return fires


def moving_average_fires(
    trace: list[float] | np.ndarray,
    threshold: float,
    *,
    window: int = 1,
    refractory: int = 0,
) -> bool:
    """Whether a per-frame score ``trace`` fires AT LEAST ONCE under the live criterion.

    The eval-path twin of the live fire decision: ``count_fires_moving_average(...) > 0``
    (moving average of the trailing ``window`` scores ≥ ``threshold``, with a post-fire
    ``refractory`` lockout). ``window <= 1`` + ``refractory <= 0`` reproduces the classic
    "any frame ≥ threshold" decision, so the default is byte-identical. Pure.
    """
    return count_fires_moving_average(trace, threshold, window=window, refractory=refractory) > 0


def count_fa_events(
    trace: list[float] | np.ndarray,
    threshold: float,
    *,
    grouping_window: int = 10,
) -> int:
    """Count DISTINCT false-accept EVENTS in a score ``trace`` (openWakeWord-style).

    The core of FA/hour (Task 2): a sustained above-threshold passage is ONE event,
    not one per frame. Consecutive at-or-above-``threshold`` frames are collapsed, and
    two separate crossings within ``grouping_window`` frames are merged into the same
    event (oWW's ``grouping_window`` so a flickering score near the boundary doesn't
    inflate the count). Pure; an empty trace / no crossing yields 0. The matching
    true-accept side (does the positive trace fire at all) uses :func:`fired_with_patience`.
    """
    arr = np.asarray(trace, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0
    gap = max(0, int(grouping_window))
    events = 0
    in_event = False
    since_last = gap + 1  # frames since the last above-threshold frame
    for score in arr:
        if score >= threshold:
            if not in_event and since_last > gap:
                events += 1  # a NEW event (far enough from the previous one)
            in_event = True
            since_last = 0
        else:
            in_event = False
            since_last += 1
    return events


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
    if is_official_wake_word(word):
        return (conf, False, "", trace)  # official words are openWakeWord-ONLY
    # Custom word, oWW did NOT fire: give the open-vocabulary KWS a shot, then the
    # few-shot enrolled detector (each gated + fully defensive — a miss leaves oWW intact).
    if getattr(cfg, "kws_enabled", True) and _kws_fires_on_clip(clip, sample_rate, word, cfg):
        return (conf, True, "kws", trace)
    if getattr(cfg, "fewshot_wake_enabled", True) and _fewshot_fires_on_clip(
        clip, sample_rate, word, cfg
    ):
        return (conf, True, "fewshot", trace)
    return (conf, False, "", trace)


def _fewshot_fires_on_clip(clip: np.ndarray, sample_rate: int, word: str, cfg: Config) -> bool:
    """Whether the few-shot enrolled detector fires on ``clip`` (the GUI/clip path).

    Loads ``word``'s saved enrolled references and replays the clip through the SAME
    rolling-window scorer + patience the live :class:`my_stt_tts.enrolled_wake.EnrolledWake`
    uses. Returns ``False`` when no references are enrolled / openWakeWord is unavailable —
    never raises. The caller has already established ``word`` is custom + enabled.
    """
    from .enrolled_wake import load_references, score_clip_enrolled

    refs = load_references(word)
    if refs is None:
        return False
    _conf, fired = score_clip_enrolled(
        clip,
        sample_rate,
        refs,
        threshold=getattr(cfg, "fewshot_threshold", 0.96),
        patience=getattr(cfg, "fewshot_patience", 2),
    )
    return bool(fired)


# --------------------------------------------------------------------------- #
# EVALUATION toolkit: positives-vs-negatives, FA/hour + ROC/DET, separation    #
# (ports openWakeWord's Apache-2.0 metrics approach — reuses score_wake_clip)   #
# --------------------------------------------------------------------------- #

# openWakeWord scores one frame per 80 ms; the live loop's FA/hour math converts a
# frame count to wall-clock with this. (1280 samples / 16 kHz = 0.08 s.)
SECONDS_PER_FRAME = FRAME_SAMPLES / 16000.0


def score_clip_set(
    clips: list[np.ndarray],
    word: str,
    *,
    sample_rate: int = 16000,
    threshold: float = 0.4,
    phases: int = 8,
    wakewords_dir: str = WAKEWORDS_DIR,
) -> tuple[list[float], list[list[float]]]:
    """Score a SET of clips for ``word`` → ``(max_scores, per_clip_traces)``.

    The shared scorer behind the histogram (Task 1) and FA-eval (Task 2): each clip is
    run through the REAL phase-diverse :func:`score_wake_clip` and reduced to its MAX
    score (the histogram value) plus its full per-frame trace (so the FA-eval can count
    EVENTS, not frames, over the negative corpus). Pure aggregation — a missing model
    or an unavailable backend makes :func:`score_wake_clip` return ``0.0`` per clip, so
    this never raises. ``sample_rate`` applies to every clip (they are 16 kHz once read
    via :func:`my_stt_tts.audio.read_wav_float`, so the default is the common case).
    """
    max_scores: list[float] = []
    traces: list[list[float]] = []
    for clip in clips:
        conf, _fired, trace = score_wake_clip(
            clip,
            sample_rate,
            word,
            threshold=threshold,
            phases=phases,
            wakewords_dir=wakewords_dir,
            with_trace=True,
        )
        max_scores.append(round(float(conf), 4))
        traces.append(trace)
    return max_scores, traces


def separation(pos_scores: list[float], neg_scores: list[float]) -> float:
    """A single separation scalar between positive and negative max-score sets.

    Higher = the word's positive clips score cleanly above the negatives (a usable
    wake word); ≈0 or negative = they overlap (the recall-vs-level problem the judge
    flagged — visible at last). Uses **d-prime** ``(μ_pos − μ_neg) / σ_pooled`` (the
    detection-theory sensitivity index) when both sides have variance; degrades to the
    plain mean gap when a side is constant (σ=0) and to ``0.0`` when either side is
    empty. Pure; rounded for a stable wire value.
    """
    pos = np.asarray(pos_scores, dtype=np.float64).ravel()
    neg = np.asarray(neg_scores, dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        return 0.0
    mean_gap = float(pos.mean() - neg.mean())
    pooled = float(np.sqrt((pos.var() + neg.var()) / 2.0))
    if pooled <= 1e-9:  # one (or both) side constant -> d-prime undefined; report the gap
        return round(mean_gap, 4)
    return round(mean_gap / pooled, 4)


def fa_eval(
    pos_traces: list[list[float]],
    neg_traces: list[list[float]],
    *,
    thresholds: list[float] | None = None,
    grouping_window: int = 10,
    window: int = 1,
    refractory: int = 0,
    target_fa: float = 0.5,
) -> dict[str, Any]:
    """Sweep ``thresholds`` → false-accepts/hour + true-accept rate (ROC/DET points).

    The operating-point curve (Task 2), computed under the SAME moving-average criterion
    the live detector uses (so the FA/hour ROC-DET numbers are TRUE OF the running loop —
    ``live == eval``). For each candidate threshold:

    * **FA / hour** — the count of LIVE fires the detector would emit on the negative
      corpus, converted to per-hour via the negatives' total wall-clock duration
      (``frames × 80 ms``). With temporal smoothing active (``window > 1`` or
      ``refractory > 0``) this is :func:`count_fires_moving_average` (the live state
      machine: moving-average crossings, refractory-spaced); with the defaults
      (``window <= 1`` + ``refractory <= 0``) it is :func:`count_fa_events`
      (consecutive above-threshold frames = ONE event, merged within ``grouping_window``)
      — byte-identical to before. Counting events / fires — NOT frames — is the whole
      point: a 3-second sustained false trigger is one annoyance, not 37.
    * **true_accept** is the fraction of POSITIVE clips that fire at the threshold under
      the same criterion (the live moving-average gate when smoothing is on, else "any
      frame at-or-above the threshold") — the recall side of the ROC/DET point.

    Returns ``{"points": [{threshold, fa_per_hour, true_accept}], "miss_at_target_fa",
    "target_fa", "neg_seconds"}``. ``miss_at_target_fa`` is the miss-rate (1 − recall)
    at ``target_fa`` false-accepts/hour, ``np.interp``-olated along the FA/hour ↦ miss
    curve (monotonized by sort), so a user can read "to stay under 0.5 FA/h you miss X%".
    Pure; an empty negative corpus yields ``fa_per_hour = 0`` everywhere and a clear
    caller-level message (the worker checks the corpus before calling).
    """
    grid = thresholds if thresholds else [round(t, 3) for t in np.linspace(0.05, 0.95, 19)]
    neg_frames = sum(len(t) for t in neg_traces)
    neg_seconds = neg_frames * SECONDS_PER_FRAME
    n_pos = len(pos_traces)
    smoothing = window > 1 or refractory > 0
    points: list[dict[str, Any]] = []
    for thr in grid:
        if smoothing:
            fa_events = sum(
                count_fires_moving_average(t, thr, window=window, refractory=refractory)
                for t in neg_traces
            )
            accepts = sum(
                1
                for t in pos_traces
                if moving_average_fires(t, thr, window=window, refractory=refractory)
            )
        else:
            fa_events = sum(
                count_fa_events(t, thr, grouping_window=grouping_window) for t in neg_traces
            )
            accepts = sum(1 for t in pos_traces if (np.asarray(t).max() if t else 0.0) >= thr)
        fa_per_hour = (fa_events / neg_seconds * 3600.0) if neg_seconds > 0 else 0.0
        true_accept = (accepts / n_pos) if n_pos else 0.0
        points.append(
            {
                "threshold": round(float(thr), 4),
                "fa_per_hour": round(float(fa_per_hour), 4),
                "true_accept": round(float(true_accept), 4),
            }
        )
    return {
        "points": points,
        "miss_at_target_fa": _miss_at_target_fa(points, target_fa),
        "target_fa": round(float(target_fa), 4),
        "neg_seconds": round(float(neg_seconds), 2),
    }


def _miss_at_target_fa(points: list[dict[str, Any]], target_fa: float) -> float:
    """Miss-rate (1 − true_accept) at ``target_fa`` FA/hour, ``np.interp``-olated.

    Builds the FA/hour ↦ miss-rate curve from the swept ``points``, sorts it by FA/hour
    (``np.interp`` requires an increasing x), and interpolates the miss-rate at the
    target false-accept budget. ``np.interp`` clamps to the endpoints outside the swept
    range, so a target below the lowest achievable FA/hour reads the strictest point's
    miss-rate. Returns ``1.0`` (total miss) when there are no points. Pure.
    """
    if not points:
        return 1.0
    fa = np.asarray([p["fa_per_hour"] for p in points], dtype=np.float64)
    miss = np.asarray([1.0 - p["true_accept"] for p in points], dtype=np.float64)
    order = np.argsort(fa)
    return round(float(np.interp(float(target_fa), fa[order], miss[order])), 4)


# --------------------------------------------------------------------------- #
# Log-mel spectrogram of a saved clip (Task 4) — scipy.signal, already installed #
# --------------------------------------------------------------------------- #


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_mels: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """A ``(n_mels, n_fft//2+1)`` triangular mel filterbank (Slaney-style, pure numpy)."""
    f_max = sample_rate / 2.0
    mel_pts = np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(f_max)), n_mels + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bins[m - 1], bins[m], bins[m + 1]
        for k in range(lo, ctr):
            if ctr > lo:
                fb[m - 1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi > ctr:
                fb[m - 1, k] = (hi - k) / (hi - ctr)
    return fb


def log_mel_spectrogram(
    clip: np.ndarray,
    sample_rate: int,
    *,
    n_mels: int = 40,
    max_frames: int = 200,
) -> dict[str, Any]:
    """Compute a downsampled log-mel spectrogram grid of ``clip`` for the GUI (Task 4).

    Resamples to 16 kHz, runs a ``scipy.signal.stft`` (25 ms window / 10 ms hop), maps
    the power spectrum through an ``n_mels``-band triangular mel filterbank, takes
    ``10·log10`` (dB), normalizes to ``[0, 1]`` for a heatmap, and **downsamples the
    time axis to at most ``max_frames`` columns** so the result event stays GUI-friendly
    (a 10 s clip is ~1000 STFT frames → binned to ≤200). Returns ``{"mels": n_mels,
    "frames": <#cols>, "grid": [[…]], "freqs": [...], "times": [...]}`` where ``grid`` is
    a ``mels × frames`` magnitude matrix (row 0 = lowest mel band), and ``freqs``/``times``
    are the band-center frequencies (Hz) and column times (s) for the axes.

    scipy is pulled in by the installed extras; if it is somehow absent this degrades to
    an empty grid rather than raising (the worker turns that into a clear message). An
    empty clip yields an empty grid.
    """
    arr = np.asarray(clip, dtype=np.float32).ravel()
    if arr.size == 0:
        return {"mels": n_mels, "frames": 0, "grid": [], "freqs": [], "times": []}
    from .audio import resample_to

    arr = resample_to(arr, int(sample_rate), 16000)
    try:
        from scipy.signal import stft
    except Exception:  # noqa: BLE001 — scipy missing -> degrade to an empty grid
        log.warning("log-mel spectrogram needs scipy; returning empty grid")
        return {"mels": n_mels, "frames": 0, "grid": [], "freqs": [], "times": []}
    n_fft = 400  # 25 ms @ 16 kHz
    hop = 160  # 10 ms @ 16 kHz
    freqs_hz, times_s, zxx = stft(
        arr, fs=16000, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False
    )
    power = np.abs(zxx) ** 2  # (n_fft//2+1, n_frames)
    fb = _mel_filterbank(n_mels, n_fft, 16000)  # (n_mels, n_fft//2+1)
    mel_power = fb @ power  # (n_mels, n_frames)
    log_mel = 10.0 * np.log10(mel_power + 1e-10)
    # Normalize to [0, 1] for a heatmap (robust to the -100 dB floor).
    lo, hi = float(log_mel.min()), float(log_mel.max())
    norm = (log_mel - lo) / (hi - lo) if hi > lo else np.zeros_like(log_mel)
    n_frames = norm.shape[1]
    # Downsample the time axis to <= max_frames columns by averaging contiguous bins.
    cols = min(max_frames, n_frames) if n_frames else 0
    if cols and n_frames > cols:
        edges = np.linspace(0, n_frames, cols + 1).astype(int)
        binned = np.stack(
            [norm[:, edges[i] : edges[i + 1]].mean(axis=1) for i in range(cols)], axis=1
        )
        col_times = np.array([float(times_s[min(edges[i], n_frames - 1)]) for i in range(cols)])
    else:
        binned = norm
        col_times = np.asarray(times_s, dtype=np.float64)
    band_centers = _mel_to_hz(
        np.linspace(_hz_to_mel(np.array(0.0)), _hz_to_mel(np.array(8000.0)), n_mels + 2)
    )[1:-1]
    return {
        "mels": int(n_mels),
        "frames": int(binned.shape[1]),
        "grid": [[round(float(v), 4) for v in row] for row in binned],
        "freqs": [round(float(f), 1) for f in band_centers],
        "times": [round(float(t), 3) for t in col_times],
    }
