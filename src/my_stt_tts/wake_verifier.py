"""Custom wake-word VERIFIER — an openWakeWord-style speaker/recall adapter (Task 3).

openWakeWord ships ``train_custom_verifier``: a tiny logistic-regression head on the
SAME shared audio embedding the base ``.onnx`` model consumes, fit to a handful of the
user's own positive clips (+ negatives) so the detector specializes to HIS voice /
room without a GPU retrain. This module ports that idea against the version-pinned
openWakeWord 0.4.0 here:

* :func:`embed_clip` pulls the 96-d shared embedding sequence for a clip out of the
  oWW ``AudioFeatures`` preprocessor (``_get_embeddings``) and **mean-pools** it to one
  per-clip feature vector — exactly what oWW's verifier averages over positive frames.
* :func:`train_verifier` fits a :class:`sklearn.linear_model.LogisticRegression` on the
  pooled positives (label 1) + negatives (label 0) and saves it (joblib) to a
  git-ignored path under ``models/wake_verifiers/<word>.joblib``.
* :class:`CustomVerifier` wraps a loaded model and scores a clip → probability, so
  :class:`my_stt_tts.wake.WakeWord` can GATE its base prediction with it.

**Hard dependency boundary:** scikit-learn + openWakeWord are BOTH optional (the
``debug`` and ``wake`` extras). Every public symbol here imports them LAZILY inside the
function body and degrades to a clear, non-raising result when they are absent, so the
core package — and ``import my_stt_tts.wake`` — stay clean without either installed.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.wake_verifier")

# Where trained verifiers are persisted. Under models/ which is git-ignored, so a
# user-specific verifier is NEVER committed (it is personal voice data, like the
# saved recordings). One file per word.
VERIFIER_DIR = "models/wake_verifiers"
# A clip must yield at least this many embedding rows to mean-pool meaningfully
# (~0.4 s of audio); shorter clips are dropped from the training set with a warning.
_MIN_EMBED_ROWS = 3


def verifier_path(word: str, *, verifier_dir: str = VERIFIER_DIR) -> str:
    """The conventional on-disk path for ``word``'s trained verifier (joblib)."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in (word or "unknown"))
    return str(Path(verifier_dir) / f"{safe or 'unknown'}.joblib")


def _embedding_model() -> Any | None:  # noqa: ANN401 — opaque oWW AudioFeatures
    """A bare openWakeWord ``AudioFeatures`` preprocessor (for ``_get_embeddings``).

    Returns ``None`` (logged) when openWakeWord is not installed — the caller degrades
    to a clear "wake extra missing" message. Constructed WITHOUT any wake model: only
    the shared melspectrogram → embedding front-end is needed.
    """
    try:
        from openwakeword.utils import AudioFeatures
    except Exception as exc:  # noqa: BLE001 — the `wake` extra isn't installed
        log.warning("custom verifier needs the openWakeWord `wake` extra: %s", exc)
        return None
    try:
        return AudioFeatures()
    except Exception as exc:  # noqa: BLE001 — model files missing / backend failure
        log.warning("could not construct openWakeWord AudioFeatures: %s", exc)
        return None


def embed_clip(
    clip: np.ndarray,
    sample_rate: int,
    *,
    features: Any = None,  # noqa: ANN401 — reuse an AudioFeatures across clips
) -> np.ndarray | None:
    """Mean-pooled 96-d openWakeWord embedding for one clip (the verifier feature).

    Resamples to 16 kHz, converts to int16 PCM (oWW's required input — see
    :func:`my_stt_tts.wake.to_int16_pcm`), runs the shared embedding model over the
    whole clip, and averages the resulting ``(N, 96)`` embedding rows to a single
    ``(96,)`` vector. Pass an existing ``features`` (``AudioFeatures``) to amortize
    construction across a training batch. Returns ``None`` when openWakeWord is
    unavailable or the clip is too short to embed (fewer than :data:`_MIN_EMBED_ROWS`
    rows) — never raises.
    """
    from .audio import resample_to
    from .wake import to_int16_pcm

    feats = features if features is not None else _embedding_model()
    if feats is None:
        return None
    arr = np.asarray(clip, dtype=np.float32).ravel()
    if arr.size == 0:
        return None
    arr = resample_to(arr, int(sample_rate), 16000)
    pcm = to_int16_pcm(arr)
    try:
        emb = np.asarray(feats._get_embeddings(pcm))  # noqa: SLF001 — oWW's documented API
    except Exception as exc:  # noqa: BLE001 — a backend/shape failure is per-clip terminal
        log.warning("embedding extraction failed: %s", exc)
        return None
    if emb.ndim != 2 or emb.shape[0] < _MIN_EMBED_ROWS:
        return None
    return emb.mean(axis=0).astype(np.float32)


def train_verifier(
    pos_clips: list[np.ndarray],
    neg_clips: list[np.ndarray],
    word: str,
    *,
    sample_rate: int = 16000,
    verifier_dir: str = VERIFIER_DIR,
) -> dict[str, Any]:
    """Train + persist a logistic-regression verifier for ``word`` (oWW-style).

    Embeds every positive (label 1) and negative (label 0) clip via :func:`embed_clip`,
    fits a :class:`sklearn.linear_model.LogisticRegression` (balanced class weights, so a
    small positive set isn't swamped by a large negative corpus), and saves it with
    joblib to :func:`verifier_path`. Returns the shared-contract dict
    ``{"trained", "path", "n_pos", "n_neg", "message"}``:

    * scikit-learn absent (``debug`` extra) → ``trained=False`` + a clear install hint.
    * fewer than 3 usable positive clips, or no usable negatives → ``trained=False`` +
      a "need ≥3 positives and ≥1 negative" message (mirrors oWW's minimum).
    * success → ``trained=True``, ``path`` set, counts of the clips that actually
      embedded (short/garbled clips are dropped).

    Never raises — every failure is reported as ``trained=False`` with a message the
    GUI shows verbatim.
    """
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # noqa: BLE001 — scikit-learn lives in the `debug` extra
        return {
            "trained": False,
            "path": "",
            "n_pos": 0,
            "n_neg": 0,
            "message": f"verifier needs scikit-learn (the `debug` extra): {exc}",
        }
    feats = _embedding_model()
    if feats is None:
        return {
            "trained": False,
            "path": "",
            "n_pos": 0,
            "n_neg": 0,
            "message": "verifier needs the openWakeWord `wake` extra (embedding model)",
        }
    pos = [
        v for v in (embed_clip(c, sample_rate, features=feats) for c in pos_clips) if v is not None
    ]
    neg = [
        v for v in (embed_clip(c, sample_rate, features=feats) for c in neg_clips) if v is not None
    ]
    if len(pos) < 3 or len(neg) < 1:
        return {
            "trained": False,
            "path": "",
            "n_pos": len(pos),
            "n_neg": len(neg),
            "message": (
                f"{word}: need >=3 positive clips and >=1 negative to train a verifier "
                f"(have {len(pos)} pos, {len(neg)} neg)"
            ),
        }
    feat_matrix = np.vstack(pos + neg)
    labels = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000)
    try:
        clf.fit(feat_matrix, labels)
    except Exception as exc:  # noqa: BLE001 — degenerate data must not crash the worker
        return {
            "trained": False,
            "path": "",
            "n_pos": len(pos),
            "n_neg": len(neg),
            "message": f"{word}: verifier training failed: {exc}",
        }
    path = verifier_path(word, verifier_dir=verifier_dir)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(clf, path)
    except Exception as exc:  # noqa: BLE001 — disk error -> report, don't crash
        return {
            "trained": False,
            "path": "",
            "n_pos": len(pos),
            "n_neg": len(neg),
            "message": f"{word}: verifier trained but could not be saved: {exc}",
        }
    return {
        "trained": True,
        "path": path,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "message": f"{word}: trained verifier on {len(pos)} pos + {len(neg)} neg -> {path}",
    }


class CustomVerifier:
    """A trained per-word verifier: ``score(clip) -> probability in [0, 1]``.

    Wraps a loaded ``LogisticRegression`` + a shared ``AudioFeatures`` preprocessor.
    :meth:`score` embeds a clip and returns the model's positive-class probability. The
    :class:`my_stt_tts.wake.WakeWord` gate calls it to confirm a base-model fire is
    really the enrolled word (``base_fired AND verifier_prob >= verifier_threshold``).
    """

    def __init__(self, clf: Any, features: Any) -> None:  # noqa: ANN401 — sklearn + oWW
        self._clf = clf
        self._features = features

    @classmethod
    def load(cls, word: str, *, verifier_dir: str = VERIFIER_DIR) -> CustomVerifier | None:
        """Load ``word``'s saved verifier, or ``None`` if absent / deps missing.

        Returns ``None`` (logged at debug) when joblib/sklearn is not installed, the
        file does not exist, or the openWakeWord embedding model can't be built — so
        :class:`my_stt_tts.wake.WakeWord` simply runs ungated. Never raises.
        """
        path = verifier_path(word, verifier_dir=verifier_dir)
        if not os.path.isfile(path):
            return None
        try:
            import joblib
        except Exception:  # noqa: BLE001 — no sklearn/joblib -> no verifier
            return None
        feats = _embedding_model()
        if feats is None:
            return None
        try:
            clf = joblib.load(path)
        except Exception as exc:  # noqa: BLE001 — corrupt/incompatible artifact
            log.warning("could not load verifier %s: %s", path, exc)
            return None
        return cls(clf, feats)

    def score(self, clip: np.ndarray, sample_rate: int = 16000) -> float:
        """Positive-class probability for ``clip`` (0.0 when it can't be embedded)."""
        vec = embed_clip(clip, sample_rate, features=self._features)
        if vec is None:
            return 0.0
        try:
            proba = float(self._clf.predict_proba(vec.reshape(1, -1))[0, 1])
        except Exception as exc:  # noqa: BLE001 — never crash a detection on the verifier
            log.warning("verifier scoring failed: %s", exc)
            return 0.0
        return proba
