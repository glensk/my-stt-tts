"""Active-learning closed loop for the wake detector (Mycroft Precise's idea, repo #5).

Mycroft Precise's training loop is: run the detector, the user flags a wrong activation, the
clip is added to the training set with the correct label, the model is retrained, and the
loop repeats — so the detector specializes to the user's voice/room over time. Precise needs
a GPU retrain per round; WE already have the CHEAP CPU rebuilders (``enroll_word`` few-shot
references + ``train_verifier`` logistic head + the per-word output ``Calibrator``, all
near-instant), plus the eval toolkit. The only thing missing was the LOOP. This module is it.

THE THREE RELABEL ACTIONS (the shared GUI contract)
---------------------------------------------------
* ``mark_false_fire`` — a clip the user marked "wasn't me": MOVE it to the per-word NEGATIVES
  dir ``debug/recordings/wake_neg/<word>/`` so the verifier + FA-eval learn it as a negative.
* ``mark_miss`` — a true wake the detector dropped: ensure the clip is a POSITIVE in
  ``debug/recordings/wake/<word>/`` so the few-shot references + verifier learn it.
* ``capture_last_fire`` — save the most-recent LIVE wake-loop fire's audio window (retained in
  a ring buffer) as a negative, then behave like ``mark_false_fire``.

THE SAFETY INTERLOCK (non-negotiable)
-------------------------------------
A mislabeled clip must NOT poison the detector. So every relabel is **eval-gated**: the
detector's positives-vs-negatives separation (d-prime) and miss-rate-at-target-FA are measured
BEFORE the rebuild; the cheap rebuilders run against a SNAPSHOT-protected copy of the model
artifacts; the gate is measured AFTER; and the rebuild is KEPT only if it does NOT regress
(d-prime not lower AND miss-at-target-FA not higher, within a tolerance). Otherwise the prior
artifacts are RESTORED (rollback) and the result reports ``accepted=False``. The user's
original enrollment is the sacrosanct FLOOR we roll back to; references are capped
(``enroll_word(max_refs=…)``); the clip move is reversible (it stays on disk under its new
label, and a rolled-back rebuild leaves the *model* untouched).

Every public entry point is fully defensive (openWakeWord / scikit-learn may be absent) and
emits a ``relabel_result`` event the GUI renders as an accepted / rolled-back card.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import Config

log = logging.getLogger("my_stt_tts.active_learning")

# The per-word active-learning NEGATIVE label folder lives alongside the positive "wake"
# training folder under the git-ignored debug/ tree (see audio.save_recording).
WAKE_NEG_KIND = "wake_neg"

# Gate tolerances: a rebuild is accepted when d-prime does not DROP by more than SEP_TOL and
# the miss-rate at the target FA/hour does not RISE by more than MISS_TOL. Small slack so a
# rebuild that is statistically flat (noise-level wiggle) is still accepted — the loop should
# make progress, not stall on rounding — while a real regression is rejected.
SEP_TOL = 1e-3
MISS_TOL = 1e-3

# The model artifacts a rebuild touches (and thus what the snapshot must protect for rollback).
# Each is a per-word file under the git-ignored models/ tree.
_ARTIFACT_KINDS = ("embeddings", "verifier", "calibration")


def neg_dir_for(word: str, *, recordings_dir: str | None = None) -> str:
    """The per-word active-learning NEGATIVES dir ``debug/recordings/wake_neg/<word>/``."""
    from .audio import _sanitize_word
    from .audio import recordings_dir as default_dir

    root = recordings_dir if recordings_dir is not None else default_dir()
    return os.path.join(root, WAKE_NEG_KIND, _sanitize_word(word or "unknown"))


def pos_dir_for(word: str, *, recordings_dir: str | None = None) -> str:
    """The per-word POSITIVES (training) dir ``debug/recordings/wake/<word>/``."""
    from .audio import _sanitize_word
    from .audio import recordings_dir as default_dir

    root = recordings_dir if recordings_dir is not None else default_dir()
    return os.path.join(root, "wake", _sanitize_word(word or "unknown"))


def load_negative_clips_union(word: str, cfg: Config) -> tuple[list[np.ndarray], list[str]]:
    """The word's negative corpus: the global ``negative_corpus_dir`` UNIONed with wake_neg/.

    The active-learning loop's negative source (repo #5): the user's shared wake-word-free
    corpus (``Config.negative_corpus_dir``) PLUS the per-word ``debug/recordings/wake_neg/<word>/``
    clips the user marked "wasn't me" / captured as live false fires. Returns
    ``(clips, dirs)`` as 16 kHz float32; an unreadable clip is skipped (logged). Pure-ish (reads
    disk); never raises.
    """
    from . import audio

    dirs = [cfg.negative_corpus_dir, neg_dir_for(word)]
    clips: list[np.ndarray] = []
    seen: set[str] = set()
    for directory in dirs:
        for path in audio.list_wavs(directory):
            real = os.path.realpath(path)
            if real in seen:
                continue
            seen.add(real)
            try:
                clip, _rate = audio.read_wav_float(path, target_rate=16000)
                clips.append(clip)
            except Exception as exc:  # noqa: BLE001 — one bad clip must not sink the set
                log.warning("skipping unreadable negative clip %s: %s", path, exc)
    return clips, dirs


def load_positive_clips(word: str) -> list[np.ndarray]:
    """Every SAVED positive clip for ``word`` as 16 kHz float32 (the eval positives)."""
    from . import audio

    clips: list[np.ndarray] = []
    for path in audio.list_wavs(pos_dir_for(word)):
        try:
            clip, _rate = audio.read_wav_float(path, target_rate=16000)
            clips.append(clip)
        except Exception as exc:  # noqa: BLE001 — one bad clip must not sink the set
            log.warning("skipping unreadable positive clip %s: %s", path, exc)
    return clips


@dataclass
class Gate:
    """The eval-gate metrics for one rebuild side: d-prime separation + miss-at-target-FA."""

    separation: float
    miss_at_target_fa: float


def compute_gate(
    word: str,
    cfg: Config,
    *,
    pos_clips: list[np.ndarray],
    neg_clips: list[np.ndarray],
    target_fa: float = 0.5,
) -> Gate:
    """Measure the current detector quality for ``word``: ``Gate(separation, miss@target_fa)``.

    Scores the positives + negatives through the REAL phase-diverse detector under the
    word's LIVE calibration (so the gate measures the running config — live == eval), then
    reduces to the two scalars the relabel loop compares before/after a rebuild:

    * ``separation`` — :func:`my_stt_tts.wake.separation` d-prime of the positive vs negative
      MAX-score distributions (higher = cleaner; the recall-vs-FA proof).
    * ``miss_at_target_fa`` — :func:`my_stt_tts.wake.fa_eval`'s miss-rate at ``target_fa``
      false-accepts/hour (lower = better recall at the FA budget).

    Empty positives → ``separation 0`` and ``miss 1.0`` (a worst-case floor); empty negatives →
    ``miss 0`` (FA is 0 everywhere). Never raises — a missing model yields the worst-case gate.
    """
    from .calibration import calibrator_for
    from .wake import fa_eval, score_clip_set, separation, wake_model_for

    if not os.path.isfile(wake_model_for(word)):
        return Gate(separation=0.0, miss_at_target_fa=1.0)
    calibrator = calibrator_for(word, enabled=getattr(cfg, "wake_calibration", False))
    pos_scores, pos_traces = (
        score_clip_set(
            pos_clips,
            word,
            threshold=cfg.wake_threshold,
            phases=cfg.wake_phases,
            calibrator=calibrator,
        )
        if pos_clips
        else ([], [])
    )
    neg_scores, neg_traces = (
        score_clip_set(
            neg_clips,
            word,
            threshold=cfg.wake_threshold,
            phases=cfg.wake_phases,
            calibrator=calibrator,
        )
        if neg_clips
        else ([], [])
    )
    sep = separation(pos_scores, neg_scores)
    fa = fa_eval(
        pos_traces,
        neg_traces,
        window=getattr(cfg, "wake_window", 1),
        refractory=getattr(cfg, "wake_refractory", 0),
        target_fa=target_fa,
    )
    return Gate(separation=float(sep), miss_at_target_fa=float(fa["miss_at_target_fa"]))


def gate_improves(before: Gate, after: Gate) -> bool:
    """Whether ``after`` does NOT regress vs ``before`` (the accept/rollback decision).

    Accepts when the d-prime separation did not DROP by more than :data:`SEP_TOL` AND the
    miss-rate at the target FA/hour did not RISE by more than :data:`MISS_TOL`. A rebuild that
    is statistically flat is accepted (the loop makes progress); a genuine regression on either
    axis is rejected (roll back). Pure.
    """
    sep_ok = after.separation >= before.separation - SEP_TOL
    miss_ok = after.miss_at_target_fa <= before.miss_at_target_fa + MISS_TOL
    return sep_ok and miss_ok


def _artifact_paths(word: str) -> dict[str, str]:
    """The per-word model-artifact paths a rebuild writes (for snapshot/rollback)."""
    from .calibration import calibration_path
    from .enrolled_wake import embeddings_path
    from .wake_verifier import verifier_path

    return {
        "embeddings": embeddings_path(word),
        "verifier": verifier_path(word),
        "calibration": calibration_path(word),
    }


def _snapshot_artifacts(word: str) -> dict[str, bytes | None]:
    """Read each model artifact's bytes (or ``None`` if absent) so a rebuild can be rolled back.

    The sacrosanct-floor mechanism: before the rebuilders touch disk we capture the EXACT prior
    state of every per-word artifact. A rolled-back relabel restores these bytes verbatim, so a
    mislabeled clip leaves the model byte-identical to before. Never raises.
    """
    snap: dict[str, bytes | None] = {}
    for kind, path in _artifact_paths(word).items():
        try:
            with open(path, "rb") as fh:
                snap[kind] = fh.read()
        except OSError:
            snap[kind] = None  # absent (or unreadable) -> rollback removes any newly-created file
    return snap


def _restore_artifacts(word: str, snapshot: dict[str, bytes | None]) -> None:
    """Restore the model artifacts to a :func:`_snapshot_artifacts` state (rollback). Defensive."""
    for kind, path in _artifact_paths(word).items():
        data = snapshot.get(kind)
        try:
            if data is None:
                if os.path.isfile(path):
                    os.remove(path)  # the artifact did not exist before -> remove the rebuild's
            else:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "wb") as fh:
                    fh.write(data)
        except OSError as exc:  # a restore failure is logged; the gate already rejected
            log.warning("rollback restore failed for %s (%s): %s", word, path, exc)


def _rebuild_detector(word: str, cfg: Config) -> bool:
    """Re-fit the cheap CPU detector for ``word`` from saved clips; return whether it rebuilt.

    Runs the three near-instant rebuilders against the CURRENT on-disk clips:

    * ``enroll_word`` — few-shot references from ``debug/recordings/wake/<word>/`` (capped).
    * ``train_verifier`` — the logistic head on positives (label 1) + the UNIONed negatives.
    * the per-word output ``Calibrator`` — re-fit from the positives' calibrated-OFF max scores
      (only persisted when ``wake_calibration`` is on AND enough samples).

    Each step is independently defensive (a missing extra simply skips that artifact). Returns
    True if at least one artifact was (re)written, so the caller knows a rebuild happened.
    """
    from .calibration import fit_and_save
    from .enrolled_wake import enroll_word
    from .wake import score_clip_set, wake_model_for
    from .wake_verifier import train_verifier

    pos_clips = load_positive_clips(word)
    neg_clips, _dirs = load_negative_clips_union(word, cfg)
    rebuilt = False

    enr = enroll_word(word)  # reads the per-word positives folder itself; caps refs
    rebuilt = rebuilt or bool(enr.get("enrolled"))

    if pos_clips and neg_clips:
        ver = train_verifier(pos_clips, neg_clips, word)
        rebuilt = rebuilt or bool(ver.get("trained"))

    # Re-fit the output calibrator from the RAW (calibration-off) positive max scores, so the
    # map reflects the freshly-labeled positive set. Only meaningful when the switch is on.
    if (
        getattr(cfg, "wake_calibration", False)
        and pos_clips
        and os.path.isfile(wake_model_for(word))
    ):
        raw_pos_scores, _ = score_clip_set(
            pos_clips, word, threshold=cfg.wake_threshold, phases=cfg.wake_phases, calibrator=None
        )
        cal = fit_and_save(word, raw_pos_scores)
        rebuilt = rebuilt or cal.enabled

    return rebuilt


@dataclass
class RelabelResult:
    """The outcome of one relabel action (mirrors the ``relabel_result`` event fields)."""

    word: str
    action: str
    rebuilt: bool
    accepted: bool
    sep_before: float
    sep_after: float
    fa_before: float
    fa_after: float
    message: str
    hash: str = ""


def rebuild_and_gate(word: str, cfg: Config, *, action: str, clip_hash: str = "") -> RelabelResult:
    """Rebuild the detector for ``word`` and KEEP it only if the eval gate does not regress.

    The safety interlock: snapshot the model artifacts, measure the gate BEFORE, run the cheap
    rebuilders, measure the gate AFTER; accept (keep) when :func:`gate_improves`, else restore
    the snapshot (roll back). Returns a :class:`RelabelResult` carrying both gates + the
    accepted flag + a human message. Never raises.
    """
    pos_clips = load_positive_clips(word)
    neg_clips, _dirs = load_negative_clips_union(word, cfg)
    before = compute_gate(word, cfg, pos_clips=pos_clips, neg_clips=neg_clips)
    snapshot = _snapshot_artifacts(word)

    rebuilt = _rebuild_detector(word, cfg)
    if not rebuilt:
        return RelabelResult(
            word=word,
            action=action,
            rebuilt=False,
            accepted=False,
            sep_before=before.separation,
            sep_after=before.separation,
            fa_before=before.miss_at_target_fa,
            fa_after=before.miss_at_target_fa,
            message=(
                f"{word}: nothing to rebuild (needs the openWakeWord/scikit-learn extras and "
                "saved clips) — clip kept under its new label"
            ),
            hash=clip_hash,
        )

    # Re-read clips AFTER the rebuild (mark_miss may have added a positive; the union is fixed
    # for this call) and measure the gate under the just-written artifacts.
    after = compute_gate(
        word,
        cfg,
        pos_clips=load_positive_clips(word),
        neg_clips=load_negative_clips_union(word, cfg)[0],
    )
    accepted = gate_improves(before, after)
    if not accepted:
        _restore_artifacts(word, snapshot)
        # Restoring the calibration file changes the calibrated gate; report the rolled-back
        # (== before) AFTER so the card shows the detector is unchanged.
        after = before
        message = (
            f"{word}: rebuild ROLLED BACK — it regressed the eval gate "
            f"(d-prime {before.separation:.3f}->{after.separation:.3f}, "
            f"miss@{0.5}FA/h {before.miss_at_target_fa:.3f}->{after.miss_at_target_fa:.3f}); "
            "golden enrollment kept"
        )
    else:
        message = (
            f"{word}: rebuild ACCEPTED — d-prime {before.separation:.3f}->{after.separation:.3f}, "
            f"miss@{0.5}FA/h {before.miss_at_target_fa:.3f}->{after.miss_at_target_fa:.3f}"
        )
    return RelabelResult(
        word=word,
        action=action,
        rebuilt=True,
        accepted=accepted,
        sep_before=before.separation,
        sep_after=after.separation,
        fa_before=before.miss_at_target_fa,
        fa_after=after.miss_at_target_fa,
        message=message,
        hash=clip_hash,
    )


def _find_clip_by_hash(clip_hash: str) -> str | None:
    """Resolve a clip's 8-hex content ``hash`` to its absolute path (``*-<hash>.wav``), or None."""
    from . import audio

    if not clip_hash:
        return None
    hits = audio.find_recordings(f"*-{clip_hash}.wav")
    return hits[0] if hits else None


def _move_clip(src: str, dest_dir: str) -> str | None:
    """Move ``src`` into ``dest_dir`` (created), returning the new path or None on error.

    Idempotent: if the file already lives in ``dest_dir`` it is left in place. The move keeps
    the original basename (so the content hash stays addressable) and is reversible — the clip
    is never deleted, only relabeled by folder.
    """
    try:
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(src))
        if os.path.realpath(src) == os.path.realpath(dest):
            return dest  # already in the right place
        shutil.move(src, dest)
        return dest
    except OSError as exc:
        log.warning("relabel move failed (%s -> %s): %s", src, dest_dir, exc)
        return None


def relabel_clip(word: str, clip_hash: str, action: str, cfg: Config) -> RelabelResult:
    """Move a clip to the right label dir for ``action`` then eval-gated-rebuild the detector.

    * ``mark_false_fire`` → move the clip into the per-word NEGATIVES dir (``wake_neg/<word>/``).
    * ``mark_miss`` → move/ensure the clip into the per-word POSITIVES dir (``wake/<word>/``).

    Then :func:`rebuild_and_gate`. A clip that can't be found / moved yields ``rebuilt=False``
    with a clear message (no rebuild attempted). Never raises.
    """
    src = _find_clip_by_hash(clip_hash)
    if src is None:
        return RelabelResult(
            word=word,
            action=action,
            rebuilt=False,
            accepted=False,
            sep_before=0.0,
            sep_after=0.0,
            fa_before=1.0,
            fa_after=1.0,
            message=f"{word}: no saved clip with hash {clip_hash!r} to relabel",
            hash=clip_hash,
        )
    dest_dir = neg_dir_for(word) if action == "mark_false_fire" else pos_dir_for(word)
    moved = _move_clip(src, dest_dir)
    if moved is None:
        return RelabelResult(
            word=word,
            action=action,
            rebuilt=False,
            accepted=False,
            sep_before=0.0,
            sep_after=0.0,
            fa_before=1.0,
            fa_after=1.0,
            message=f"{word}: could not move clip {clip_hash!r} to {dest_dir}",
            hash=clip_hash,
        )
    return rebuild_and_gate(word, cfg, action=action, clip_hash=clip_hash)


def capture_last_fire(word: str, clip: np.ndarray, sample_rate: int, cfg: Config) -> RelabelResult:
    """Save the last live wake fire's audio ``clip`` as a NEGATIVE for ``word``, then relabel.

    The ``capture_last_fire`` action: the live wake loop retains the last fire's audio window
    in a ring buffer; this saves it via :func:`my_stt_tts.audio.save_recording` with
    ``kind="wake_neg"`` (→ ``debug/recordings/wake_neg/<word>/``) and then behaves like
    ``mark_false_fire`` (eval-gated rebuild). An empty clip yields ``rebuilt=False`` with a
    clear message. Never raises.
    """
    from . import audio

    arr = np.asarray(clip, dtype=np.float32).ravel()
    if arr.size == 0:
        return RelabelResult(
            word=word,
            action="capture_last_fire",
            rebuilt=False,
            accepted=False,
            sep_before=0.0,
            sep_after=0.0,
            fa_before=1.0,
            fa_after=1.0,
            message=f"{word}: no recent live fire audio to capture",
        )
    _path, hash8, _url = audio.save_recording(
        arr, int(sample_rate), kind=WAKE_NEG_KIND, source="server", word=word
    )
    return rebuild_and_gate(word, cfg, action="capture_last_fire", clip_hash=hash8)


def emit_relabel_result(result: RelabelResult) -> None:
    """Publish a :class:`RelabelResult` on the event bus as a ``relabel_result`` event."""
    from .events import bus

    bus.relabel_result(
        word=result.word,
        action=result.action,
        rebuilt=result.rebuilt,
        accepted=result.accepted,
        sep_before=result.sep_before,
        sep_after=result.sep_after,
        fa_before=result.fa_before,
        fa_after=result.fa_after,
        message=result.message,
        hash=result.hash,
    )
