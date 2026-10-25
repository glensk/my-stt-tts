"""Pick the voice-check threshold and margin from scored clips (tp#1235; pure, no I/O).

``scripts/calibrate_voice.py`` scores every clip it has against Albert's call profile and
hands them here as :class:`Sample` objects:

* positives — Albert's enrollment takes (scored leave-one-out, so a take never judges
  itself) and in-call clips labelled ``albert``;
* negatives — in-call clips labelled ``other`` and synthetic text-to-speech voices.

The bridge accepts an utterance when ``score ≥ threshold`` and, with the margin rule on,
``score − best_negative ≥ margin`` (``best_negative``: the highest cosine against any
negative clip — see ``bridge.Authoriser``). :func:`recommend` aims for no false accepts on
the negatives while at least ``target`` (default 95%) of Albert's SHORT sentences that are
long enough to authorise anything (≥ the minimum length) still pass:

1. ``t_hi`` = the highest threshold that keeps ``target`` of the short positives;
   ``t_lo`` = the best negative score + ``safety``. If ``t_lo ≤ t_hi`` the threshold sits in
   the middle of that gap; otherwise it is ``t_hi`` (Albert first) and only the margin can
   separate the rest.
2. The margin is recommended only once at least ``min_others`` real ``other`` clips exist
   (synthetic voices alone say little about a family member). It sits between the largest
   negative margin + ``safety`` and the largest margin that still keeps ``target`` of the
   short positives — or at the latter when the two cross.

:attr:`Recommendation.feasible` says whether both goals were met; when not, the outcome
lists how many negatives would still pass.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

#: Below this many short positives (long enough to count) nothing is recommended.
MIN_SHORT_POSITIVES = 8
DEFAULT_TARGET = 0.95
DEFAULT_MIN_OTHERS = 10
DEFAULT_SAFETY = 0.02
#: Speech shorter than this counts as a "short" sentence for in-call clips (enrollment takes
#: carry their own short / long tag).
SHORT_S = 2.5
_EPS = 1e-9  # float noise in score differences must not flip a decision


@dataclass(frozen=True)
class Sample:  # pylint: disable=too-many-instance-attributes  # one scored clip
    """One scored clip (see module docstring)."""

    source: str  # "enrolled", "labelled" or "synthetic"
    positive: bool
    score: float
    best_negative: float | None
    seconds: float
    short: bool
    name: str = ""

    @property
    def gap(self) -> float | None:
        """``score − best_negative`` (None without negatives)."""
        return None if self.best_negative is None else self.score - self.best_negative


@dataclass
class Outcome:  # pylint: disable=too-many-instance-attributes  # one evaluation
    """How one setting does on the samples (only positives ≥ the minimum length count)."""

    threshold: float
    margin: float | None
    short_accepted: int = 0
    short_total: int = 0
    accepted: int = 0
    total: int = 0
    false_accepts: int = 0
    negatives: int = 0
    false_accept_names: list[str] = field(default_factory=list)

    @property
    def short_rate(self) -> float:
        return self.short_accepted / self.short_total if self.short_total else float("nan")


@dataclass
class Recommendation:
    """What :func:`recommend` suggests, and how it does."""

    threshold: float
    margin: float | None
    outcome: Outcome
    feasible: bool
    others: int
    notes: list[str] = field(default_factory=list)


def accepts(sample: Sample, threshold: float, margin: float | None) -> bool:
    """Would the bridge accept this clip (threshold, then the margin when one is set)?"""
    if sample.score < threshold - _EPS:
        return False
    gap = sample.gap
    return margin is None or gap is None or gap >= margin - _EPS


def evaluate(
    samples: Sequence[Sample], threshold: float, margin: float | None, min_s: float
) -> Outcome:
    """Count accepted positives (short ones separately) and false accepts."""
    out = Outcome(threshold, margin)
    for sample in samples:
        ok = accepts(sample, threshold, margin)
        if not sample.positive:
            out.negatives += 1
            if ok:
                out.false_accepts += 1
                out.false_accept_names.append(sample.name or sample.source)
            continue
        if sample.seconds < min_s:
            continue  # refused by length anyway; the speaker check never decides it
        out.total += 1
        out.accepted += ok
        if sample.short:
            out.short_total += 1
            out.short_accepted += ok
    return out


def _floor2(value: float) -> float:
    return math.floor(value * 100 + 1e-9) / 100


def _ceil2(value: float) -> float:
    return math.ceil(value * 100 - 1e-9) / 100


def _between(low: float, high: float) -> float:
    """A two-decimal value in the middle of [low, high] that stays inside it."""
    lo, hi = _ceil2(low), _floor2(high)
    if lo > hi:
        return round((low + high) / 2, 4)
    return min(max(round((low + high) / 2, 2), lo), hi)


def _keep_target(values: list[float], target: float) -> tuple[float, int]:
    """(highest cut keeping ``target`` of ``values`` at ≥ cut, rejects that allows)."""
    ordered = sorted(values)
    allowed = math.floor((1.0 - target) * len(ordered) + 1e-9)
    return ordered[allowed], allowed


def recommend(  # pylint: disable=too-many-arguments,too-many-locals
    samples: Sequence[Sample],
    *,
    min_s: float,
    target: float = DEFAULT_TARGET,
    min_others: int = DEFAULT_MIN_OTHERS,
    safety: float = DEFAULT_SAFETY,
) -> Recommendation | None:
    """The recommended threshold / margin; None with too few short positives (see module)."""
    short_pos = [s for s in samples if s.positive and s.short and s.seconds >= min_s]
    negatives = [s for s in samples if not s.positive]
    others = sum(1 for s in negatives if s.source == "labelled")
    if len(short_pos) < MIN_SHORT_POSITIVES:
        return None
    notes: list[str] = []
    t_hi, allowed = _keep_target([s.score for s in short_pos], target)
    t_lo = max((s.score for s in negatives), default=-1.0) + safety
    if t_lo <= t_hi:
        threshold = _between(t_lo, t_hi)
        needs_margin = False
    else:
        threshold = _floor2(t_hi)
        needs_margin = True
        notes.append(
            f"the threshold alone cannot separate them: the best negative scores"
            f" {t_lo - safety:.2f}, {target:.0%} of the short sentences need ≤ {t_hi:.2f}"
        )
    margin = None
    if others >= min_others:
        margin = _margin(short_pos, negatives, threshold, allowed, safety)
        if margin is None:
            notes.append("no margin possible: the negatives have no comparison clips")
    elif negatives:
        notes.append(f"margin rule off: {others} labelled 'other' clip(s), it needs {min_others}")
    outcome = evaluate(samples, threshold, margin, min_s)
    feasible = outcome.false_accepts == 0 and outcome.short_rate >= target - 1e-9
    if needs_margin and margin is None:
        notes.append("label more 'other' clips (label_voice_clips.py -p) to enable the margin")
    return Recommendation(threshold, margin, outcome, feasible, others, notes)


def _margin(
    short_pos: list[Sample],
    negatives: list[Sample],
    threshold: float,
    allowed: int,
    safety: float,
) -> float | None:
    """The margin between the negatives' best gap and what the short positives allow."""
    passing = [s for s in short_pos if s.score >= threshold and s.gap is not None]
    left = allowed - sum(1 for s in short_pos if s.score < threshold)
    gaps_pos = sorted(s.gap for s in passing if s.gap is not None)
    gaps_neg = [s.gap for s in negatives if s.gap is not None and s.score >= threshold]
    all_neg = [s.gap for s in negatives if s.gap is not None]
    if not gaps_pos or not all_neg:
        return None
    m_hi = gaps_pos[max(left, 0)] if left < len(gaps_pos) else gaps_pos[-1]
    m_lo = max(gaps_neg or all_neg) + safety
    if m_lo <= m_hi:
        return _between(m_lo, m_hi)
    return _floor2(m_hi)


__all__ = [
    "DEFAULT_MIN_OTHERS",
    "DEFAULT_SAFETY",
    "DEFAULT_TARGET",
    "MIN_SHORT_POSITIVES",
    "SHORT_S",
    "Outcome",
    "Recommendation",
    "Sample",
    "accepts",
    "evaluate",
    "recommend",
]
