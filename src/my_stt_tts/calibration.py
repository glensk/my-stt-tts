"""Per-word OUTPUT calibration of the wake score (Mycroft Precise's ``ThresholdDecoder`` idea).

Mycroft Precise post-processes its raw network output through a ``ThresholdDecoder``: it
models the score distribution and maps the raw value through that distribution's CDF, so the
DECISION threshold means the same thing regardless of how the underlying model happens to
scale its outputs. Different wake models (and even the same model on different words) put the
"clearly the word" region at wildly different raw scores — openWakeWord's maziko peaks ~0.67
while an official word saturates near 1.0 — so a single ``wake_threshold`` can never be right
for all of them at once. Calibration fixes the SCALE, not the trigger logic.

This ports the IDEA (NOT Precise's GRU): a per-word monotone map

    calibrated(raw) = Φ((logit(raw) − μ) / σ)

fit from the saved POSITIVE clips' max-score statistics (reused via the eval toolkit's
``score_clip_set`` / ``separation``). ``Φ`` is the standard-normal CDF, ``logit`` the
log-odds. The map is:

* **monotone increasing** in ``raw`` (``logit`` and ``Φ`` are both monotone), so it never
  re-orders detections — a louder/clearer utterance always scores at least as high;
* **model-independent at 0.5** — ``μ`` is centered on the LOW edge of the positive cluster
  (``mean(logit) − CENTER_SIGMA · std(logit)``), so a calibrated score of 0.5 corresponds to
  a raw score at the bottom of the user's own genuine utterances. ``threshold = 0.5`` then
  means "as confident as your weakest real wake" on EVERY word, instead of a raw number that
  only made sense for one model;
* **default identity** — :func:`Calibrator.identity` (and any insufficiently-fit calibrator)
  returns ``raw`` unchanged, so calibration OFF is byte-identical to the un-calibrated path.

The fitted ``(mu, sigma, n)`` parameters are persisted PER WORD next to the embedding/verifier
artifacts under the git-ignored ``models/`` tree (one ``.json`` per word — they are derived
from the user's personal voice clips). :class:`my_stt_tts.wake.WakeWord` applies the map to
the per-frame MAX-over-phases score BEFORE the moving-average deque, AND
:func:`my_stt_tts.wake.score_wake_clip` applies the IDENTICAL map to the same per-frame score —
so the offline eval still measures the TRUE live behaviour (``live == eval``).
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.calibration")

# Where per-word calibration parameters are persisted. Under models/ which is git-ignored,
# so a user-specific calibration (derived from his own voice clips) is NEVER committed — like
# the enrolled references and trained verifiers. One .json per word.
CALIBRATION_DIR = "models/wake_calibration"

# A calibration needs at least this many usable positive max-scores to fit a meaningful
# (mu, sigma); below it the map stays identity (calibration silently OFF for that word).
MIN_CALIBRATION_SAMPLES = 5

# Where calibrated 0.5 lands relative to the positive cluster: mu is placed CENTER_SIGMA
# standard deviations BELOW the positive logit mean, so 0.5 corresponds to "as confident as
# the low edge of your genuine wakes" (most real utterances then calibrate above 0.5). One
# sigma down is the conventional lower-shoulder of a normal cluster.
CENTER_SIGMA = 1.0

# Numerical guards: logit saturates at 0/1, so raw scores are clamped into this open interval
# before the log-odds, and a degenerate (constant) positive set floors sigma here so the map
# stays finite (and effectively a steep-but-monotone step at mu).
_EPS = 1e-6
_MIN_SIGMA = 1e-3


def calibration_path(word: str, *, calibration_dir: str = CALIBRATION_DIR) -> str:
    """The conventional on-disk path for ``word``'s calibration parameters (``.json``)."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in (word or "unknown"))
    return str(Path(calibration_dir) / f"{safe or 'unknown'}.json")


def _logit(p: np.ndarray | float) -> np.ndarray:
    """Log-odds of ``p``, clamped into ``(_EPS, 1-_EPS)`` so 0/1 don't blow up to ±inf."""
    arr = np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(arr / (1.0 - arr))


def _phi(z: float) -> float:
    """Standard-normal CDF via :func:`math.erf` (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class Calibrator:
    """A per-word monotone score map ``calibrated(raw) = Φ((logit(raw) − μ) / σ)``.

    Construct with :meth:`fit` from saved positive-clip max scores, :meth:`load` from a saved
    ``.json``, or :meth:`identity` for the no-op pass-through. :meth:`apply` maps one raw score
    (or an array of them) into ``[0, 1]``. An identity calibrator (``enabled is False``)
    returns its input unchanged, so the calibration-OFF path is byte-identical.
    """

    def __init__(self, mu: float | None, sigma: float | None, *, n: int = 0) -> None:
        # mu/sigma None => identity (the no-op map). Otherwise the fitted log-odds center +
        # spread of the positive cluster. n is the sample count the fit used (for the GUI).
        self.mu = mu
        self.sigma = sigma
        self.n = int(n)

    @property
    def enabled(self) -> bool:
        """True when this calibrator applies a real (non-identity) map."""
        return self.mu is not None and self.sigma is not None

    @classmethod
    def identity(cls) -> Calibrator:
        """The pass-through calibrator: ``apply(raw) == raw`` (calibration OFF / insufficient)."""
        return cls(None, None, n=0)

    @classmethod
    def fit(
        cls,
        pos_scores: list[float] | np.ndarray,
        *,
        min_samples: int = MIN_CALIBRATION_SAMPLES,
        center_sigma: float = CENTER_SIGMA,
    ) -> Calibrator:
        """Fit a calibrator from the POSITIVE clips' max scores, or identity if too few.

        ``pos_scores`` is the per-clip MAX wake score over every saved positive clip for the
        word (exactly what :func:`my_stt_tts.wake.score_clip_set` returns). With fewer than
        ``min_samples`` usable values this returns :meth:`identity` (calibration stays OFF for
        the word). Otherwise ``μ`` is the positive logit mean shifted ``center_sigma`` standard
        deviations DOWN (so calibrated 0.5 sits at the low edge of genuine wakes) and ``σ`` is
        the positive logit spread (floored at :data:`_MIN_SIGMA`). Pure; never raises.
        """
        arr = np.asarray(pos_scores, dtype=np.float64).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size < int(min_samples):
            return cls.identity()
        logits = _logit(arr)
        spread = float(logits.std())
        sigma = max(spread, _MIN_SIGMA)
        mu = float(logits.mean()) - float(center_sigma) * sigma
        return cls(mu, sigma, n=int(arr.size))

    def apply(self, raw: float | np.ndarray) -> Any:  # noqa: ANN401 — mirrors input type
        """Map a raw score (scalar or array) through the calibration; identity when disabled.

        Monotone increasing in ``raw`` and clamped to ``[0, 1]`` via ``Φ``. A scalar in yields
        a Python ``float`` out; an array in yields a same-shaped ``np.ndarray`` — so it drops
        straight into the per-frame score path without changing the caller's types.
        """
        if not self.enabled:
            return raw
        assert self.mu is not None and self.sigma is not None  # narrowed by .enabled
        is_scalar = np.isscalar(raw) or (isinstance(raw, np.ndarray) and raw.ndim == 0)
        logits = _logit(raw)
        z = (logits - self.mu) / self.sigma
        if is_scalar:
            return float(_phi(float(z)))
        vec = np.asarray(z, dtype=np.float64)
        # Vectorized Φ via erf (math.erf is scalar-only); 0.5*(1+erf(z/√2)).
        out = 0.5 * (1.0 + np.vectorize(math.erf)(vec / math.sqrt(2.0)))
        return out.astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Serializable parameters (for persistence + the GUI ``wake_word_info`` state)."""
        return {"mu": self.mu, "sigma": self.sigma, "n": self.n, "enabled": self.enabled}

    def save(self, word: str, *, calibration_dir: str = CALIBRATION_DIR) -> str:
        """Persist this calibrator to ``word``'s ``.json``; returns the path ("" on disk error).

        An identity calibrator is NOT written (nothing to persist) — the absence of a file IS
        the off state. Never raises; a disk error is logged and yields an empty path.
        """
        if not self.enabled:
            return ""
        path = calibration_path(word, calibration_dir=calibration_dir)
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"word": word, **self.to_dict()}, fh, indent=2)
        except OSError as exc:
            log.warning("calibration save failed (%s): %s", path, exc)
            return ""
        return path

    @classmethod
    def load(cls, word: str, *, calibration_dir: str = CALIBRATION_DIR) -> Calibrator:
        """Load ``word``'s saved calibration, or :meth:`identity` when absent / unreadable.

        Returns the identity (no-op) calibrator on any failure — a missing file, corrupt JSON,
        or missing/degenerate parameters — so the live path simply runs un-calibrated. Never
        raises.
        """
        path = calibration_path(word, calibration_dir=calibration_dir)
        if not os.path.isfile(path):
            return cls.identity()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            mu = data.get("mu")
            sigma = data.get("sigma")
            if mu is None or sigma is None:
                return cls.identity()
            return cls(float(mu), float(sigma), n=int(data.get("n", 0)))
        except (OSError, ValueError, TypeError) as exc:
            log.warning("could not load calibration %s: %s", path, exc)
            return cls.identity()


def fit_and_save(
    word: str,
    pos_scores: list[float] | np.ndarray,
    *,
    calibration_dir: str = CALIBRATION_DIR,
    min_samples: int = MIN_CALIBRATION_SAMPLES,
) -> Calibrator:
    """Fit a calibrator for ``word`` from ``pos_scores`` and persist it (the rebuild entry point).

    A convenience wrapper around :meth:`Calibrator.fit` + :meth:`Calibrator.save`: when the fit
    is real (enough samples) it is written to disk; an identity fit (too few samples) is
    returned without writing. Never raises.
    """
    cal = Calibrator.fit(pos_scores, min_samples=min_samples)
    cal.save(word, calibration_dir=calibration_dir)
    return cal


def calibrator_for(
    word: str,
    *,
    enabled: bool,
    calibration_dir: str = CALIBRATION_DIR,
) -> Calibrator:
    """The calibrator the LIVE/eval path should use for ``word`` given the ``enabled`` switch.

    When ``enabled`` is False (config ``wake_calibration`` off) this is ALWAYS the identity map
    (byte-identical, no disk read). When enabled, it loads ``word``'s saved parameters — and if
    none exist (never fit, or too few samples) it is still the identity map. So calibration only
    ever changes behaviour when BOTH the switch is on AND a real fit was persisted.
    """
    if not enabled:
        return Calibrator.identity()
    return Calibrator.load(word, calibration_dir=calibration_dir)
