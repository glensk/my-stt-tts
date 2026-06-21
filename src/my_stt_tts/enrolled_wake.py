"""Few-shot ENROLLED wake detector (EfficientWord-Net's idea, empirically adapted).

EfficientWord-Net enrolls a handful of the user's own clips of a word and fires on the
MAX cosine similarity of live audio to those enrolled references — no GPU retrain, the
few-shot path openWakeWord lacks. This ports the IDEA against the embedding ALREADY in
the process: openWakeWord's shared 96-d audio embedding (the same one
:mod:`my_stt_tts.wake_verifier` uses), so it adds **zero new model dependency** — we do
NOT pull EWN's 88 MB contrastively-trained ResNet-50 ONNX.

WHY MEAN-POOL, NOT WINDOWING (the empirical gate, see ``PLAN_wake_checker_loop.md``)
-----------------------------------------------------------------------------------
EWN's ResNet emits per-window embeddings that are each contrastively discriminative, so it
slides a window and takes the max cosine over windows. The oWW shared embedding is NOT like
that: measured leave-one-out on Albert's saved ``maziko`` clips, taking the max cosine over
individual ~775 ms oWW embedding ROWS lets a spurious negative frame align too well (max-neg
up to 0.99 > every positive) and DESTROYS separation (d-prime 0.80, negative margin). The
discriminative signal lives in the **whole-utterance MEAN** of the rows — mean-pooling gives
d-prime 5.41 whole-clip / 2.52 in the live rolling regime, with a clean threshold. So this
detector MEAN-POOLS the embedding rows of each rolling window (NOT max-over-rows), then scores
the window's mean-vector by MAX cosine to the per-clip enrolled references.

FALSE-ACCEPT CONTROL
--------------------
The streaming margin is real but tight (one hard negative — another wake-word attempt — can
graze the lowest positive). So the detector is conservative: it fires only after ``patience``
CONSECUTIVE rolling windows clear the threshold (the same de-bouncing :mod:`my_stt_tts.wake`
applies to oWW). The operating threshold is tuned against a NEGATIVES set, not positives alone.

INTEGRATION
-----------
:class:`EnrolledWake` exposes the EXACT :class:`my_stt_tts.wake.WakeWord` surface
(``detect``/``reset``/``last_score`` + ``threshold``/``model_name``/``available``) so the
wake loop drives it identically, and it is OR-combined as a THIRD branch in
:class:`my_stt_tts.wake.OrCombinedWake` for CUSTOM words only — official words
(hey_jarvis/alexa/hey_mycroft) stay byte-identical (the ``is_official_wake_word`` guardrail,
mirroring KWS).

HARD DEPENDENCY BOUNDARY
------------------------
openWakeWord is the optional ``wake`` extra. Every public symbol lazy-imports it inside the
function body and degrades to a clear, non-raising result when it is absent, so the core
package — and ``import my_stt_tts.enrolled_wake`` — stay clean without it installed.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np

log = logging.getLogger("my_stt_tts.enrolled_wake")

# Where the per-word enrolled reference embeddings are persisted. Under models/ which is
# git-ignored, so a user-specific enrollment is NEVER committed (it is personal voice data,
# like the saved recordings and the trained verifiers). One .npz per word.
EMBEDDINGS_DIR = "models/wake_embeddings"

# Rolling-window geometry for the live detector. The enrolled references are WHOLE-clip
# (~2 s) mean-pools of a spoken word, so the live window must be long enough that its own
# mean-pool resembles them: a window-length sweep on Albert's maziko clips (leave-one-out,
# thr 0.95) showed recall climbing 0%@1.0 s → 83%@1.5 s → 100%@1.75 s with a POSITIVE margin
# (min-pos 0.9682 > max-neg 0.9556) — a too-short window dilutes the word and collapses the
# score. So the window is ~1.75 s, advanced ~0.25 s per step. The oWW embedding front-end
# needs ≥ ~0.8 s (76 mel frames) to yield even one embedding row, so a window shorter than
# that is skipped (it would raise inside oWW and embed to nothing).
WINDOW_SECONDS = 1.75
HOP_SECONDS = 0.25
_MIN_WINDOW_SECONDS = 0.8
SAMPLE_RATE = 16000


def embeddings_path(word: str, *, embeddings_dir: str = EMBEDDINGS_DIR) -> str:
    """The conventional on-disk path for ``word``'s enrolled references (``.npz``)."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in (word or "unknown"))
    return str(Path(embeddings_dir) / f"{safe or 'unknown'}.npz")


def _l2(vec: np.ndarray) -> np.ndarray:
    """L2-normalize a 1-D vector (identity on a zero vector)."""
    norm = float(np.linalg.norm(vec))
    return vec if norm == 0.0 else (vec / norm).astype(np.float32)


def mean_embed(
    clip: np.ndarray,
    sample_rate: int,
    *,
    features: Any = None,  # noqa: ANN401 — reuse an oWW AudioFeatures across clips
) -> np.ndarray | None:
    """Mean-pooled, L2-normalized 96-d oWW embedding for ``clip`` (the reference vector).

    Resamples to 16 kHz, converts to int16 PCM (oWW's required input), runs the shared
    embedding model over the audio and **averages** the ``(N, 96)`` rows to one ``(96,)``
    vector, L2-normalized for cosine. This is the SAME aggregation
    :func:`my_stt_tts.wake_verifier.embed_clip` uses — re-implemented here L2-normalized
    so it doubles as the enrollment vector AND the live window score. Returns ``None`` when
    openWakeWord is unavailable or the clip is too short to embed — never raises.
    """
    from .wake_verifier import _embedding_model, embed_clip

    feats = features if features is not None else _embedding_model()
    if feats is None:
        return None
    vec = embed_clip(clip, sample_rate, features=feats)
    if vec is None:
        return None
    return _l2(np.asarray(vec, dtype=np.float32))


def enrolled_clips_for(
    word: str,
    *,
    recordings_dir: str | None = None,
) -> list[str]:
    """Saved clip paths to enroll ``word`` from: the per-word folder + loose labelled WAVs.

    Wake tests are saved as training data under ``debug/recordings/wake/<word>/*.wav`` (see
    :func:`my_stt_tts.audio.save_recording`); older / loose captures land flat in
    ``debug/recordings/`` with the word in the filename (``*-<word>-*.wav``). This returns
    BOTH, de-duplicated + sorted, so an enrollment uses every available sample of the word.
    ``recordings_dir`` defaults to :func:`my_stt_tts.audio.recordings_dir`.
    """
    from .audio import _sanitize_word
    from .audio import recordings_dir as default_dir

    root = recordings_dir if recordings_dir is not None else default_dir()
    safe = _sanitize_word(word or "unknown")
    hits: set[str] = set()
    hits.update(glob.glob(os.path.join(root, "wake", safe, "*.wav")))
    hits.update(glob.glob(os.path.join(root, f"*-{word}-*.wav")))
    return sorted(hits)


def enroll_word(
    word: str,
    *,
    clips: list[np.ndarray] | None = None,
    sample_rate: int = SAMPLE_RATE,
    recordings_dir: str | None = None,
    embeddings_dir: str = EMBEDDINGS_DIR,
    min_clips: int = 3,
) -> dict[str, Any]:
    """Enroll ``word`` from clips → per-clip reference embeddings saved to ``.npz``.

    Source of clips: the explicit ``clips`` list when given, else every saved sample of the
    word found by :func:`enrolled_clips_for`. Each clip is mean-pooled + L2-normalized via
    :func:`mean_embed`; the references are stored PER-CLIP (NOT averaged into one centroid —
    the max-cosine detector compares against each, which the Phase-1 analysis validated) to
    :func:`embeddings_path`. Returns the shared-contract dict
    ``{"enrolled", "path", "n_refs", "message"}``:

    * openWakeWord absent (``wake`` extra) → ``enrolled=False`` + a clear install hint.
    * fewer than ``min_clips`` usable clips → ``enrolled=False`` + a "need >= N" message.
    * success → ``enrolled=True``, ``path`` set, ``n_refs`` = clips that actually embedded.

    Never raises — every failure is reported as ``enrolled=False`` with a message a CLI / GUI
    shows verbatim, mirroring :func:`my_stt_tts.wake_verifier.train_verifier`.
    """
    from .audio import read_wav_float
    from .wake_verifier import _embedding_model

    feats = _embedding_model()
    if feats is None:
        return {
            "enrolled": False,
            "path": "",
            "n_refs": 0,
            "message": f"{word}: enrollment needs the openWakeWord `wake` extra (embedding model)",
        }
    audio_clips: list[tuple[np.ndarray, int]] = []
    if clips is not None:
        audio_clips = [(np.asarray(c, dtype=np.float32), sample_rate) for c in clips]
    else:
        for path in enrolled_clips_for(word, recordings_dir=recordings_dir):
            try:
                audio_clips.append(read_wav_float(path, target_rate=SAMPLE_RATE))
            except (OSError, ValueError) as exc:  # skip an unreadable/empty clip
                log.warning("enroll: skipping unreadable clip %s: %s", path, exc)
    refs = [
        v for v in (mean_embed(c, sr, features=feats) for c, sr in audio_clips) if v is not None
    ]
    if len(refs) < min_clips:
        return {
            "enrolled": False,
            "path": "",
            "n_refs": len(refs),
            "message": (
                f"{word}: need >= {min_clips} usable clips to enroll "
                f"(have {len(refs)}; record more via scripts/enroll_wakeword.py)"
            ),
        }
    matrix = np.vstack(refs).astype(np.float32)
    path = embeddings_path(word, embeddings_dir=embeddings_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, refs=matrix, word=np.asarray(word))
    except OSError as exc:
        return {
            "enrolled": False,
            "path": "",
            "n_refs": len(refs),
            "message": f"{word}: enrolled but could not be saved: {exc}",
        }
    return {
        "enrolled": True,
        "path": path,
        "n_refs": len(refs),
        "message": f"{word}: enrolled {len(refs)} reference clips -> {path}",
    }


def load_references(
    word: str,
    *,
    embeddings_dir: str = EMBEDDINGS_DIR,
) -> np.ndarray | None:
    """Load ``word``'s saved ``(n_refs, 96)`` reference matrix, or ``None`` if absent.

    Returns ``None`` (logged at debug) when the file does not exist or is unreadable, so the
    caller cleanly falls back to no enrolled detector. The stored references are already
    L2-normalized (see :func:`enroll_word`); re-normalized defensively on load. Never raises.
    """
    path = embeddings_path(word, embeddings_dir=embeddings_dir)
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as data:
            refs = np.asarray(data["refs"], dtype=np.float32)
    except (OSError, ValueError, KeyError) as exc:  # corrupt / incompatible artifact
        log.warning("could not load enrolled references %s: %s", path, exc)
        return None
    if refs.ndim != 2 or refs.shape[0] == 0:
        return None
    norms = np.linalg.norm(refs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (refs / norms).astype(np.float32)


def score_window(
    window: np.ndarray,
    references: np.ndarray,
    features: Any = None,  # noqa: ANN401 — reuse an oWW AudioFeatures across windows
) -> float:
    """MAX cosine of ``window``'s mean-pooled embedding to any enrolled reference (0..1).

    The per-window score the live detector thresholds: mean-pool the window's oWW embedding
    rows, L2-normalize, and return the maximum dot-product against the (L2-normalized)
    reference matrix. ``0.0`` when the window can't be embedded (too short / oWW missing).
    """
    vec = mean_embed(window, SAMPLE_RATE, features=features)
    if vec is None or references.size == 0:
        return 0.0
    return float(np.max(references @ vec))


class EnrolledWake:
    """Few-shot enrolled wake detector with the :class:`my_stt_tts.wake.WakeWord` surface.

    Holds the per-clip enrolled reference embeddings for one word and a rolling audio buffer.
    Each :meth:`detect` appends the incoming frame, and once the buffer holds a full window it
    scores the window (mean-pool → max cosine to references). A fire requires ``patience``
    CONSECUTIVE windows at-or-above ``threshold`` (the false-accept de-bounce). ``last_score``
    is the most recent window score so the debug instrument can plot it. On any embedding
    failure it degrades to never-firing (``detect`` → ``False``), never raising.
    """

    def __init__(
        self,
        references: np.ndarray,
        word: str,
        *,
        threshold: float = 0.96,
        patience: int = 2,
        window_seconds: float = WINDOW_SECONDS,
        hop_seconds: float = HOP_SECONDS,
    ) -> None:
        self.references = np.asarray(references, dtype=np.float32)
        self.word = word
        self.model_name = word
        self.threshold = float(threshold)
        self.patience = max(1, int(patience))
        self._window_samples = int(window_seconds * SAMPLE_RATE)
        self._hop_samples = max(1, int(hop_seconds * SAMPLE_RATE))
        self._min_samples = int(_MIN_WINDOW_SECONDS * SAMPLE_RATE)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._since_score = 0  # samples accumulated since the last window was scored
        self._consecutive = 0  # consecutive windows at-or-above threshold (for patience)
        self.last_score: float = 0.0
        self._features: Any = None  # lazily-built shared oWW AudioFeatures
        self._broken = False

    @classmethod
    def from_config(cls, cfg: Any, word: str | None = None) -> EnrolledWake | None:  # noqa: ANN401
        """Build the enrolled detector for ``word`` (default ``cfg.wake_phrase``), or ``None``.

        Returns ``None`` — enrollment disabled, an OFFICIAL word (the guardrail), or no saved
        references — so the caller cleanly runs without it. Mirrors
        :meth:`my_stt_tts.kws.SherpaKws.from_config`. Fully defensive: never raises.
        """
        from .config import is_official_wake_word

        target = word or cfg.wake_phrase
        if not getattr(cfg, "fewshot_wake_enabled", True):
            return None
        if is_official_wake_word(target):
            return None  # official words stay openWakeWord-only (byte-identical)
        refs = load_references(target)
        if refs is None:
            return None
        log.info("wake: loaded %d enrolled references for %r", refs.shape[0], target)
        return cls(
            refs,
            target,
            threshold=getattr(cfg, "fewshot_threshold", 0.96),
            patience=getattr(cfg, "fewshot_patience", 2),
        )

    def available(self) -> bool:
        """True if this detector has at least one enrolled reference embedding."""
        return self.references.ndim == 2 and self.references.shape[0] > 0

    def _ensure_features(self) -> bool:
        """Build (once) the shared oWW embedding front-end; latch broken on failure."""
        if self._broken:
            return False
        if self._features is None:
            from .wake_verifier import _embedding_model

            self._features = _embedding_model()
            if self._features is None:
                self._broken = True
                return False
        return True

    def detect(self, frame: np.ndarray) -> bool:
        """Return ``True`` if the enrolled word fired after appending this audio ``frame``.

        Accepts the SAME native float32 16 kHz frame the wake loop feeds the other detectors
        (no int16 conversion at this boundary — :func:`mean_embed` converts at the oWW edge).
        Buffers the frame; once ``hop_seconds`` of new audio has arrived AND the buffer holds a
        usable window, scores the most recent window (mean-pool → max cosine to references) and
        updates :attr:`last_score`. A fire requires ``patience`` consecutive windows at-or-above
        :attr:`threshold`. Never raises — an embedding failure latches the detector to
        never-fire.
        """
        if self._broken or not self.available() or not self._ensure_features():
            return False
        arr = np.asarray(frame, dtype=np.float32).ravel()
        self._buffer = np.concatenate([self._buffer, arr])
        # Keep only the most recent window worth of audio (the rolling buffer).
        if self._buffer.size > self._window_samples:
            self._buffer = self._buffer[-self._window_samples :]
        self._since_score += arr.size
        # Score at most once per hop, and only with enough audio to embed.
        if self._since_score < self._hop_samples or self._buffer.size < self._min_samples:
            return False
        self._since_score = 0
        try:
            score = score_window(self._buffer, self.references, features=self._features)
        except Exception as exc:  # noqa: BLE001 — a per-window failure latches never-fire
            log.warning("enrolled-wake scoring failed for %r: %s", self.word, exc)
            self._broken = True
            return False
        self.last_score = score
        if score >= self.threshold:
            self._consecutive += 1
            if self._consecutive >= self.patience:
                self._consecutive = 0  # consume the run so it must rebuild to re-fire
                return True
        else:
            self._consecutive = 0
        return False

    def reset(self) -> None:
        """Clear the rolling buffer + patience run between activations (fresh listen)."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._since_score = 0
        self._consecutive = 0
        self.last_score = 0.0


@overload
def score_clip_enrolled(
    clip: np.ndarray,
    sample_rate: int,
    references: np.ndarray,
    *,
    threshold: float = ...,
    patience: int = ...,
    features: Any = ...,
    with_trace: Literal[False] = ...,
) -> tuple[float, bool]: ...


@overload
def score_clip_enrolled(
    clip: np.ndarray,
    sample_rate: int,
    references: np.ndarray,
    *,
    threshold: float = ...,
    patience: int = ...,
    features: Any = ...,
    with_trace: Literal[True],
) -> tuple[float, bool, list[float]]: ...


def score_clip_enrolled(
    clip: np.ndarray,
    sample_rate: int,
    references: np.ndarray,
    *,
    threshold: float = 0.96,
    patience: int = 2,
    features: Any = None,  # noqa: ANN401 — reuse an oWW AudioFeatures across a batch
    with_trace: bool = False,
) -> tuple[float, bool] | tuple[float, bool, list[float]]:
    """Score a recorded ``clip`` against enrolled ``references`` (the offline / eval path).

    Resamples to 16 kHz and slides the SAME rolling window the live :class:`EnrolledWake`
    uses (``WINDOW_SECONDS`` / ``HOP_SECONDS``), mean-pooling each window and taking the max
    cosine to any reference — so the offline number matches what the running loop would see.
    Returns ``(confidence, fired)`` where ``confidence`` is the MAX window score over the clip
    and ``fired`` requires ``patience`` consecutive windows at-or-above ``threshold`` (via
    :func:`my_stt_tts.wake.fired_with_patience`). ``with_trace=True`` adds the per-window score
    trace (the input to the eval toolkit's :func:`my_stt_tts.wake.fa_eval`). Defensive — an
    empty clip / unembeddable audio yields zero confidence rather than raising.
    """
    from .audio import resample_to
    from .wake import fired_with_patience
    from .wake_verifier import _embedding_model

    def _result(conf: float, fired: bool, trace: list[float]) -> Any:  # noqa: ANN401
        return (conf, fired, trace) if with_trace else (conf, fired)

    refs = np.asarray(references, dtype=np.float32)
    arr = resample_to(np.asarray(clip, dtype=np.float32).ravel(), int(sample_rate), SAMPLE_RATE)
    if arr.size == 0 or refs.ndim != 2 or refs.shape[0] == 0:
        return _result(0.0, False, [])
    norms = np.linalg.norm(refs, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    refs = (refs / norms).astype(np.float32)
    feats = features if features is not None else _embedding_model()
    if feats is None:
        return _result(0.0, False, [])
    win = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop = max(1, int(HOP_SECONDS * SAMPLE_RATE))
    min_samples = int(_MIN_WINDOW_SECONDS * SAMPLE_RATE)
    trace: list[float] = []
    for start in range(0, max(1, arr.size - min_samples + 1), hop):
        seg = arr[start : start + win]
        if seg.size < min_samples:
            continue
        trace.append(round(score_window(seg, refs, features=feats), 4))
    if not trace:
        return _result(0.0, False, [])
    best = max(trace)
    fired = fired_with_patience(trace, threshold, patience=patience)
    return _result(best, fired, trace)
