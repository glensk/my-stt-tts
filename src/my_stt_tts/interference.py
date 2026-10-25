"""Keep in-call speech the voice check refused or the agent ignored, to train on later (tp#1235).

mac-voice decides from the voice alone whether a sentence is Albert's. Other people talking
during a call (family, TV, a guest) are exactly the voices that check must learn to refuse,
and the only place to hear them is a real call. :class:`CallCapture` keeps, per call:

* every utterance the bridge's :class:`~my_stt_tts.bridge.Authoriser` refused as
  ``other_speaker`` (decision ``other_speaker``);
* every utterance the agent did not answer — it called ``skip_turn`` because the speech was
  not addressed to it, so no spoken reply followed within :data:`REPLY_WINDOW_S`
  (``skipped``);
* a random share (``MAC_VOICE_INTERFERENCE_SAMPLE``, default 0.2) of the utterances that
  passed the voice check (``verified``) or that the agent answered (``answered``) — the
  positives to compare against.

Clips go to ``debug/recordings/interference/YYYY-MM/<id>.wav`` with a ``<id>.json`` next to
it: decision, cosine score against the authorised profile and which profile (``call`` /
``wake``), seconds, the time, and a label once one is set. No transcript text is stored.
Files are 0600, folders 0700, and ``debug/`` is gitignored. ``scripts/label_voice_clips.py``
plays clips and labels them ``albert`` or ``other`` (no names); ``scripts/calibrate_voice.py``
then uses ``albert`` clips as positives and ``other`` clips as negatives.

Unlabelled clips are deleted after ``MAC_VOICE_INTERFERENCE_DAYS`` (default 90) — checked
at every call start and by ``label_voice_clips.py -e``; labelled clips stay until deleted.
``MAC_VOICE_INTERFERENCE_CAPTURE=0`` turns the capture off. All saving and scoring runs on
one background thread, so the call's audio and the agent's tools never wait for it, and a
failure only logs one line (never content).
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import random
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .bridge import _env_float
from .voice_gate import REPO_ROOT
from .voice_profiles import SAMPLE_RATE, private_dir, read_json, write_json, write_wav

log = logging.getLogger("my_stt_tts.interference")

ROOT = REPO_ROOT / "debug" / "recordings" / "interference"
CAPTURE_ENV = "MAC_VOICE_INTERFERENCE_CAPTURE"
SAMPLE_ENV = "MAC_VOICE_INTERFERENCE_SAMPLE"
DAYS_ENV = "MAC_VOICE_INTERFERENCE_DAYS"
DEFAULT_SAMPLE = 0.2
DEFAULT_DAYS = 90.0
LABELS = ("albert", "other")
DECISIONS = ("other_speaker", "skipped", "verified", "answered")
#: Shorter utterances are not kept (nothing to learn a voice from).
MIN_CLIP_S = 0.5
#: At most this many clips per call (a long call with the TV on must not fill the disk).
MAX_CLIPS_PER_CALL = 40
#: A spoken agent reply this soon after a transcript answers it; later, the turn was skipped.
REPLY_WINDOW_S = 10.0
_OFF = frozenset({"0", "off", "no", "false"})


@dataclass(frozen=True)
class Config:
    """Capture settings (``from_env`` reads the ``MAC_VOICE_INTERFERENCE_*`` variables)."""

    enabled: bool = True
    sample: float = DEFAULT_SAMPLE
    days: float = DEFAULT_DAYS
    root: Path = ROOT

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, root: Path = ROOT) -> Config:
        env = os.environ if env is None else env
        enabled = (env.get(CAPTURE_ENV) or "1").strip().casefold() not in _OFF
        sample = _env_float(env, SAMPLE_ENV, DEFAULT_SAMPLE)
        days = _env_float(env, DAYS_ENV, DEFAULT_DAYS)
        return cls(enabled, min(sample, 1.0), days, root)


# -- the clip folder ---------------------------------------------------------------------------
@dataclass
class Clip:
    """One kept clip: ``<id>.wav`` + ``<id>.json`` (its metadata)."""

    wav: Path
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.wav.stem

    @property
    def sidecar(self) -> Path:
        return self.wav.with_suffix(".json")

    @property
    def label(self) -> str | None:
        value = self.meta.get("label")
        return value if value in LABELS else None

    @property
    def decision(self) -> str:
        return str(self.meta.get("decision", ""))

    @property
    def score(self) -> float | None:
        value = self.meta.get("score")
        return float(value) if isinstance(value, int | float) else None

    @property
    def duration(self) -> float:
        value = self.meta.get("duration")
        return float(value) if isinstance(value, int | float) else 0.0

    def created(self) -> datetime:
        """When it was kept (from the metadata; the file time when that is missing)."""
        raw = self.meta.get("created")
        if isinstance(raw, str):
            with contextlib.suppress(ValueError):
                return datetime.fromisoformat(raw)
        return datetime.fromtimestamp(self.wav.stat().st_mtime).astimezone()


class ClipStore:
    """The interference clip folder: save, list, find, label, delete, expire."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root

    def save(  # pylint: disable=too-many-arguments
        self,
        pcm: np.ndarray,
        *,
        decision: str,
        score: float | None,
        profile: str | None,
        now: datetime | None = None,
    ) -> Clip:
        """Keep ``pcm`` (16 kHz float32) with its metadata; returns the new clip."""
        now = (now or datetime.now()).astimezone()
        clip_id = f"{now:%Y%m%d-%H%M%S}-{decision}-{secrets.token_hex(3)}"
        folder = private_dir(self.root / f"{now:%Y-%m}")
        pcm = np.asarray(pcm, dtype=np.float32).ravel()
        meta: dict[str, Any] = {
            "version": 1,
            "created": now.isoformat(timespec="seconds"),
            "decision": decision,
            "score": None if score is None else round(float(score), 4),
            "profile": profile,
            "duration": round(pcm.size / SAMPLE_RATE, 2),
            "sample_rate": SAMPLE_RATE,
            "label": None,
        }
        clip = Clip(folder / f"{clip_id}.wav", meta)
        write_wav(clip.wav, pcm)
        write_json(clip.sidecar, meta)
        return clip

    def clips(self) -> list[Clip]:
        """Every clip with both files, oldest first."""
        out = []
        for sidecar in sorted(self.root.glob("*/*.json")):
            wav = sidecar.with_suffix(".wav")
            meta = read_json(sidecar)
            if wav.is_file() and isinstance(meta, dict):
                out.append(Clip(wav, meta))
        return out

    def find(self, ref: str) -> Clip | None:
        """The clip whose id is ``ref`` (or the only one starting with it, or that path)."""
        clips = self.clips()
        exact = [c for c in clips if ref in (c.id, str(c.wav), str(c.sidecar))]
        if exact:
            return exact[0]
        prefixed = [c for c in clips if c.id.startswith(ref)]
        return prefixed[0] if len(prefixed) == 1 else None

    def set_label(self, clip: Clip, label: str | None, now: datetime | None = None) -> Clip:
        """Label a clip ``albert`` / ``other``, or remove its label (``None``)."""
        if label is not None and label not in LABELS:
            raise ValueError(f"label must be one of {', '.join(LABELS)}")
        clip.meta["label"] = label
        clip.meta["labelled"] = (
            None
            if label is None
            else (now or datetime.now()).astimezone().isoformat(timespec="seconds")
        )
        write_json(clip.sidecar, clip.meta)
        return clip

    def delete(self, clip: Clip) -> None:
        """Remove both files (and the month folder once it is empty)."""
        for path in (clip.wav, clip.sidecar):
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
        with contextlib.suppress(OSError):
            clip.wav.parent.rmdir()  # only succeeds when empty

    def expire(self, days: float = DEFAULT_DAYS, now: datetime | None = None) -> int:
        """Delete unlabelled clips older than ``days``; returns how many."""
        cutoff = (now or datetime.now()).astimezone() - timedelta(days=days)
        gone = 0
        for clip in self.clips():
            if clip.label is None and clip.created() < cutoff:
                self.delete(clip)
                gone += 1
        return gone


# -- per-call capture --------------------------------------------------------------------------
@dataclass
class _Turn:
    """A user transcript whose outcome (answered / skipped) is not known yet."""

    received_at: float
    utterances: list[Any]


class CallCapture:  # pylint: disable=too-many-instance-attributes  # per-call state + worker
    """Decides per transcript which utterances to keep (see module docstring)."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        store: ClipStore,
        config: Config,
        *,
        scorer: Any = None,
        authorized: str | None = None,
        rand: Callable[[], float] = random.random,
        clock: Callable[[], float] = time.monotonic,
        background: bool = True,
    ) -> None:
        self.store = store
        self.config = config
        self.scorer = scorer  # voice_gate.VoiceGate: scores clips that have no score yet
        self.authorized = authorized
        self.rand = rand
        self.clock = clock
        self.background = background
        self._lock = threading.Lock()
        self._pending: dict[int, _Turn] = {}
        self._kept: set[tuple[float, float]] = set()
        self._count = 0
        self._jobs: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    # -- events (any thread; never raise) ---------------------------------------------------
    def begin_call(self) -> None:
        """A call starts: forget the last one, delete expired clips in the background."""
        with self._lock:
            self._pending.clear()
            self._kept.clear()
            self._count = 0
        self._submit(self._expire)

    def end_call(self) -> None:
        """The call ended: every transcript still without a reply was skipped."""
        with self._lock:
            turns = list(self._pending.values())
            self._pending.clear()
        for turn in turns:
            self._keep_all(turn.utterances, "skipped")

    def on_transcript(self, transcript: Any) -> None:
        """A user transcript bound to its utterance(s)."""
        if getattr(transcript, "injected", False):
            return
        utterances = [u for u in _candidates(transcript.binding) if u.duration >= MIN_CLIP_S]
        stale = self._take_stale()
        with self._lock:
            if utterances:
                self._pending[transcript.seq] = _Turn(transcript.received_at, utterances)
        for turn in stale:
            self._keep_all(turn.utterances, "skipped")

    def on_agent_said(self, text: str) -> None:
        """The agent's reply text: spoken → recent turns were answered; ``...`` → skipped."""
        spoken = bool(str(text).strip(" .…\n\t"))
        now = self.clock()
        with self._lock:
            turns = list(self._pending.values())
            self._pending.clear()
        for turn in turns:
            if spoken and now - turn.received_at <= REPLY_WINDOW_S:
                self._maybe_keep(turn.utterances, "answered")
            else:
                self._keep_all(turn.utterances, "skipped")

    def on_verdict(self, transcript: Any, code: str | None, scored: list[tuple[Any, Any]]) -> None:
        """The Authoriser's decision on a transcript: ``code`` None = verified."""
        with self._lock:
            self._pending.pop(int(getattr(transcript, "seq", -1)), None)
        if code == "other_speaker":
            for utt, score in scored:
                self._keep(utt, "other_speaker", score)
        elif code is None and scored and self.rand() < self.config.sample:
            for utt, score in scored:
                self._keep(utt, "verified", score)

    def flush(self, timeout: float = 10.0) -> None:
        """Wait until every queued save has run (tests, shutdown)."""
        if self._worker is None:
            return
        done = threading.Event()
        self._jobs.put(done.set)
        done.wait(timeout)

    def shutdown(self) -> None:
        """Finish the queued saves and stop the worker thread."""
        if self._worker is not None:
            self.flush()
            self._jobs.put(None)

    # -- deciding -----------------------------------------------------------------------------
    def _take_stale(self) -> list[_Turn]:
        """Pending turns older than the reply window: the agent never answered them."""
        now = self.clock()
        with self._lock:
            stale = [s for s, t in self._pending.items() if now - t.received_at > REPLY_WINDOW_S]
            return [self._pending.pop(s) for s in stale]

    def _maybe_keep(self, utterances: list[Any], decision: str) -> None:
        if utterances and self.rand() < self.config.sample:
            self._keep_all(utterances, decision)

    def _keep_all(self, utterances: list[Any], decision: str) -> None:
        for utt in utterances:
            self._keep(utt, decision, None)

    def _keep(self, utt: Any, decision: str, score: Any) -> None:
        if not self.config.enabled or utt is None or utt.duration < MIN_CLIP_S:
            return
        key = (round(float(utt.start), 3), round(float(utt.end), 3))
        with self._lock:
            if key in self._kept or self._count >= MAX_CLIPS_PER_CALL:
                return
            self._kept.add(key)
            self._count += 1
        pcm = np.array(utt.pcm, dtype=np.float32, copy=True)
        value = float(score) if isinstance(score, int | float) else None
        self._submit(lambda: self._save(pcm, decision, value))

    # -- the worker -----------------------------------------------------------------------------
    def _submit(self, job: Callable[[], None]) -> None:
        if not self.config.enabled:
            return
        if not self.background:
            self._run(job)
            return
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._loop, name="interference-capture", daemon=True
            )
            self._worker.start()
        self._jobs.put(job)

    def _loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._run(job)

    @staticmethod
    def _run(job: Callable[[], None]) -> None:
        try:
            job()
        except Exception:  # pylint: disable=broad-exception-caught  # never break the call
            log.warning("⚠️  interference capture failed", exc_info=True)

    def _save(self, pcm: np.ndarray, decision: str, score: float | None) -> None:
        profile = None
        if self.scorer is not None and self.authorized:
            if score is None:
                score = self.scorer.score_against(pcm, self.authorized)
            profile = _profile_kind(self.scorer, self.authorized)
        clip = self.store.save(pcm, decision=decision, score=score, profile=profile)
        shown = "-" if score is None else f"{score:.2f}"
        log.info("🎞️  kept a %s clip (%s, %.1f s)", decision, shown, clip.duration)

    def _expire(self) -> None:
        gone = self.store.expire(self.config.days)
        if gone:
            log.info(
                "🧹 deleted %d unlabelled interference clip(s) older than %g days",
                gone,
                self.config.days,
            )


def _profile_kind(scorer: Any, name: str) -> str | None:
    """``call`` / ``wake``: which profile the scorer judged ``name`` on (None if unknown)."""
    kind_of = getattr(scorer, "profile_kind", None)
    if not callable(kind_of):
        return None
    kind = kind_of(name)
    return kind if isinstance(kind, str) else None


def _candidates(binding: Any) -> list[Any]:
    if binding is None:
        return []
    candidates = getattr(binding, "candidates", None)
    return list(candidates) if candidates is not None else [binding]


def attach(
    controller: Any, scorer: Any, env: Mapping[str, str] | None = None
) -> CallCapture | None:
    """Give the bridge controller a :class:`CallCapture` (None when switched off)."""
    config = Config.from_env(env)
    if not config.enabled:
        log.info("🎞️  interference capture off (%s=0)", CAPTURE_ENV)
        return None
    authoriser = controller.authoriser
    capture = CallCapture(
        ClipStore(config.root), config, scorer=scorer, authorized=authoriser.authorized
    )
    authoriser.observer = capture.on_verdict
    controller.capture = capture
    controller.on_call(capture.begin_call, capture.end_call)
    controller.on_shutdown(capture.shutdown)
    log.info(
        "🎞️  keeping refused / ignored in-call speech in %s (unlabelled for %g days)",
        config.root,
        config.days,
    )
    return capture


__all__ = [
    "CAPTURE_ENV",
    "DAYS_ENV",
    "DECISIONS",
    "DEFAULT_DAYS",
    "LABELS",
    "SAMPLE_ENV",
    "CallCapture",
    "Clip",
    "ClipStore",
    "Config",
    "attach",
]
