"""Live speaker-ID glue: turn a recorded utterance clip into an enrolled name (G7).

The matching math and the ECAPA embedder live in :mod:`my_stt_tts.speaker_id`;
this module is the thin, *gated, fully-defensive* bridge that the runtime loops
actually call. It exists because the speaker-ID pieces were unit-tested but never
invoked in the live turn path — so per-speaker memory in :mod:`my_stt_tts.memory`
never keyed to a real person.

:class:`SpeakerPipeline` is constructed **once** per process and only becomes
active when ALL of:

* ``cfg.speaker_id_enabled`` is set (opt-in), AND
* enrolled per-person centroids exist under ``cfg.enroll_dir``, AND
* ``speechbrain`` is importable (the ``speaker`` extra is installed).

If any precondition fails, :meth:`SpeakerPipeline.from_config` returns ``None`` —
the loops then never embed and the speaker stays ``None`` (the common case: most
users have no enrollment, so there is no model load and no latency hit). Even when
active, :meth:`identify` wraps the embed + match in ``try/except`` so a corrupt
clip or a model failure degrades to ``None`` rather than crashing a turn.

Enrollment format (``cfg.enroll_dir``): one ``<name>.npy`` per enrolled person,
each a saved centroid embedding (1-D float array). ``<name>.npz`` with a
``centroid`` array is also accepted. A directory ``<name>/`` containing multiple
``*.npy`` embeddings is averaged into a centroid. Files named ``unknown`` /
``ambiguous`` / ``_guest`` are ignored (those are reserved result keys).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .memory import GUEST_KEY
from .speaker_id import AMBIGUOUS, UNKNOWN, EcapaEmbedder, SpeakerIdentifier

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from .config import Config
    from .diarize import SherpaDiarizer

log = logging.getLogger("my_stt_tts.speaker_pipeline")

# A named diarized segment: (start_seconds, end_seconds, enrolled_name | None). A
# ``None`` name is a guest / unrecognized voice / background TV (rejected by the
# ECAPA threshold) — the loop drops these for command routing.
NamedSegment = tuple[float, float, "str | None"]

# Result names that must never be read as an enrolled person's centroid file.
_RESERVED = frozenset({UNKNOWN, AMBIGUOUS, GUEST_KEY})


def load_centroids(enroll_dir: Path) -> dict[str, np.ndarray]:
    """Load per-person enrollment centroids from ``enroll_dir`` (best-effort).

    Returns ``{name: centroid}``. Missing dir / unreadable files are skipped (a
    warning is logged) rather than raised — enrollment is optional. A ``<name>/``
    sub-directory of ``*.npy`` embeddings is averaged into one centroid.
    """
    centroids: dict[str, np.ndarray] = {}
    if not enroll_dir.is_dir():
        return centroids
    for entry in sorted(enroll_dir.iterdir()):
        name = entry.stem if entry.is_file() else entry.name
        if name in _RESERVED or name.startswith("."):
            continue
        try:
            vec = _load_one(entry)
        except Exception as exc:  # noqa: BLE001 — a bad enrollment file must not crash startup
            log.warning("skipping enrollment %s: %s", entry.name, exc)
            continue
        if vec is not None and vec.size:
            centroids[name] = vec
    return centroids


def _load_one(entry: Path) -> np.ndarray | None:
    """Load a single enrollment entry into a 1-D centroid (or ``None`` to skip)."""
    if entry.is_dir():
        embeddings = [
            np.asarray(np.load(f), dtype=np.float32).ravel() for f in sorted(entry.glob("*.npy"))
        ]
        return np.mean(embeddings, axis=0) if embeddings else None
    if entry.suffix == ".npy":
        return np.asarray(np.load(entry), dtype=np.float32).ravel()
    if entry.suffix == ".npz":
        with np.load(entry) as data:
            key = "centroid" if "centroid" in data else next(iter(data.files), None)
            return np.asarray(data[key], dtype=np.float32).ravel() if key else None
    return None


class SpeakerPipeline:
    """Embed a clip and resolve it to an enrolled name (or ``None``) for the loop (G7).

    Holds the heavy :class:`~my_stt_tts.speaker_id.EcapaEmbedder` (lazy SpeechBrain
    load) and a :class:`~my_stt_tts.speaker_id.SpeakerIdentifier` built from the
    enrolled centroids + the configured thresholds. The runtime calls
    :meth:`identify` once per spoken utterance, immediately before ``brain.stream``,
    and feeds the result to ``brain.set_speaker`` so conversation memory is
    per-person. ``unknown`` / ``ambiguous`` results are mapped to ``None`` (which
    :func:`my_stt_tts.memory.speaker_key` buckets as a shared guest) so an
    unrecognized voice never reads/writes an enrolled person's history.
    """

    def __init__(
        self,
        identifier: SpeakerIdentifier,
        *,
        embedder: EcapaEmbedder | None = None,
        sample_rate: int = 16000,
        diarizer: SherpaDiarizer | None = None,
    ) -> None:
        self.identifier = identifier
        self.embedder = embedder or EcapaEmbedder()
        self.sample_rate = sample_rate
        # When set (diarization enabled + sherpa + models present), a turn is split
        # into per-speaker segments BEFORE ECAPA naming (see :meth:`identify_segments`).
        self.diarizer = diarizer

    @classmethod
    def from_config(cls, cfg: Config) -> SpeakerPipeline | None:
        """Build the pipeline, or return ``None`` if speaker-ID is not usable.

        Gated and defensive: returns ``None`` (no model load, no latency) when
        disabled, when no enrolled centroids exist, or when ``speechbrain`` is not
        importable. Construction itself is wrapped so a broken install degrades to
        ``None`` instead of crashing the loop.

        Within-turn diarization (G7+) is a STRICT superset: when
        ``cfg.speaker_diarize_enabled`` is also set AND a sherpa diarizer is buildable
        (sherpa-onnx + the models present), the pipeline gains per-segment splitting;
        otherwise the diarizer is ``None`` and the pipeline behaves single-speaker as
        before. Diarization can NEVER be active without speaker ID (it reuses the
        ECAPA naming path), so it is wired here.
        """
        if not getattr(cfg, "speaker_id_enabled", False):
            return None
        try:
            centroids = load_centroids(Path(cfg.enroll_dir))
            if not centroids:
                log.info("speaker ID enabled but no enrolled voices in %s", cfg.enroll_dir)
                return None
            import importlib.util

            if importlib.util.find_spec("speechbrain") is None:
                log.warning("speaker ID enabled but `speechbrain` is not installed; skipping")
                return None
            identifier = SpeakerIdentifier.from_config(cfg, centroids)
            log.info("speaker ID active: %d enrolled (%s)", len(centroids), ", ".join(centroids))
            # Diarizer is itself gated + defensive: None unless diarization is enabled
            # AND sherpa + the models are usable (it never crashes the build).
            from .diarize import SherpaDiarizer

            diarizer = SherpaDiarizer.from_config(cfg)
            return cls(
                identifier,
                sample_rate=getattr(cfg, "sample_rate", 16000),
                diarizer=diarizer,
            )
        except Exception as exc:  # noqa: BLE001 — never let speaker-ID setup break the loop
            log.warning("speaker ID disabled (setup failed): %s", exc)
            return None

    def identify(self, clip: np.ndarray | None) -> str | None:
        """Resolve a recorded clip to an enrolled name, or ``None`` (guest/failure).

        Defensive end to end: an empty/None clip, an embed failure, or a match that
        lands on ``unknown`` / ``ambiguous`` all return ``None`` so the caller falls
        back to the shared guest bucket. Never raises — a turn must not die because
        speaker ID hiccuped.
        """
        if clip is None or not isinstance(clip, np.ndarray) or clip.size == 0:
            return None
        return self._embed_match(np.asarray(clip, dtype=np.float32).ravel())

    def _embed_match(self, audio: np.ndarray) -> str | None:
        """Embed one audio slice and resolve it to an enrolled name, or ``None``.

        The shared core of :meth:`identify` and :meth:`identify_segments`: runs the
        EXISTING ECAPA embed + :class:`SpeakerIdentifier` match, mapping
        ``unknown`` / ``ambiguous`` (and any failure) to ``None``. So background TV /
        overlapped speech that scores below ``speaker_threshold`` becomes ``None``.
        """
        if audio.size == 0:
            return None
        try:
            embedding = self.embedder.embed(audio, sample_rate=self.sample_rate)
            name = self.identifier.identify(embedding)
        except Exception as exc:  # noqa: BLE001 — degrade to guest, keep the turn alive
            log.warning("speaker ID failed for this segment: %s", exc)
            return None
        if name in (UNKNOWN, AMBIGUOUS):
            return None
        return name

    def identify_segments(self, clip: np.ndarray | None) -> list[NamedSegment]:
        """Diarize ``clip`` into per-speaker segments, each NAMED via the ECAPA path.

        When a diarizer is wired (diarization enabled + sherpa + models present), the
        clip is split into anonymous speaker segments; each segment's audio slice is
        embedded with the SAME :class:`~my_stt_tts.speaker_id.EcapaEmbedder` and matched
        with the SAME :class:`~my_stt_tts.speaker_id.SpeakerIdentifier` as single-speaker
        ID — so ``enroll/*.npy`` centroids + the calibrated threshold apply unchanged.
        Background TV / overlapped speech falls below ``speaker_threshold`` and is
        labelled ``None`` (the caller drops these for command routing).

        Returns ``[(start_s, end_s, name | None)]`` sorted by start time. Fully
        defensive: a ``None``/empty clip -> ``[]``; with no diarizer (off / unavailable)
        the WHOLE clip is one segment named by the existing single-speaker path, so the
        result is a drop-in superset of :meth:`identify` — never raises.
        """
        if clip is None or not isinstance(clip, np.ndarray) or clip.size == 0:
            return []
        audio = np.asarray(clip, dtype=np.float32).ravel()
        total_s = audio.size / float(self.sample_rate)
        if self.diarizer is None:
            # Single-speaker fallback: the whole clip is one segment, named exactly as
            # the legacy :meth:`identify` would name it.
            return [(0.0, total_s, self.identify(audio))]
        try:
            segments = self.diarizer.segments(audio)
        except Exception as exc:  # noqa: BLE001 — diarizer must never crash a turn
            log.warning("diarization failed; single-speaker fallback: %s", exc)
            return [(0.0, total_s, self.identify(audio))]
        named: list[NamedSegment] = []
        for start_s, end_s, _local in segments:
            lo = max(0, int(round(start_s * self.sample_rate)))
            hi = min(audio.size, int(round(end_s * self.sample_rate)))
            slice_audio = audio[lo:hi] if hi > lo else np.zeros(0, dtype=np.float32)
            named.append((start_s, end_s, self._embed_match(slice_audio)))
        return named

    @property
    def diarize_active(self) -> bool:
        """True when within-turn diarization is wired (a diarizer is present)."""
        return self.diarizer is not None


@dataclass
class TurnSpeakers:
    """Outcome of resolving who spoke in a captured turn (G7 / G7+).

    ``routed`` is the enrolled person the command + memory key to (or ``None`` for
    guest). ``should_drop`` is True when the turn is ONLY unknown / background voices
    (TV) and so must NOT drive a command. ``diarized`` is True when within-turn
    diarization split the turn into >1 segment and emitted its OWN per-segment
    ``speaker`` events — the caller then emits per-known-segment LABELLED transcripts
    (via :meth:`emit_transcripts`) instead of the single plain one.
    """

    routed: str | None = None
    should_drop: bool = False
    diarized: bool = False
    segments: list[NamedSegment] = field(default_factory=list)

    def emit_transcripts(self, text: str, *, bus: Any, source: str = "live_audio") -> bool:
        """Emit a speaker-LABELLED transcript per KNOWN diarized segment; True if it did.

        parakeet returns no per-word timestamps, so the full ``text`` is attributed to
        each known speaker (segment-level attribution, as designed) rather than split
        word-by-word. Each known segment publishes a transcript tagged with its
        enrolled name (rendered ``[albert] …``) via the ``speaker`` field. Returns
        True when diarization owned the transcript (so the caller skips the plain
        one); False for the single-speaker path (caller emits the plain transcript).
        """
        if not self.diarized:
            return False
        for _start, _end, name in self.segments:
            if name is not None:
                bus.transcript(text, source=source, speaker=name)
        return True


def resolve_turn_speaker(
    speaker_id: SpeakerPipeline | None,
    clip: np.ndarray | None,
    *,
    bus: Any,
) -> TurnSpeakers:
    """Resolve a captured turn to a routing speaker, emitting per-segment bus events.

    THE single entry point the live loops call to wire diarization into a turn:
    embeds + names the clip (single-speaker) or, when within-turn diarization is
    active, splits it into per-speaker segments and names each via the EXISTING ECAPA
    path, emitting one ``bus.speaker`` per segment. Returns a :class:`TurnSpeakers`
    (routing speaker, drop verdict, whether it diarized, the named segments).

    Defensive: no pipeline / no clip / diarization off all return the legacy
    single-speaker shape (one ``bus.speaker``, never a drop) — never raises.
    """
    if speaker_id is None or clip is None:
        bus.speaker(None)
        return TurnSpeakers()
    if not getattr(speaker_id, "diarize_active", False):
        # Legacy single-speaker path: one identify + one speaker event.
        name = speaker_id.identify(clip)
        bus.speaker(name)
        return TurnSpeakers(routed=name)
    segments = speaker_id.identify_segments(clip)
    if len(segments) <= 1:
        # Diarizer collapsed to one segment (short clip / fallback): treat as the
        # legacy single-speaker turn so the caller emits the plain transcript.
        name = segments[0][2] if segments else None
        bus.speaker(name)
        return TurnSpeakers(routed=name)
    # Multi-speaker turn: emit ONE speaker event per segment for the UI.
    for _start, _end, name in segments:
        bus.speaker(name)
    routed, should_drop = route_speaker(segments)
    return TurnSpeakers(routed=routed, should_drop=should_drop, diarized=True, segments=segments)


def route_speaker(segments: list[NamedSegment]) -> tuple[str | None, bool]:
    """Decide the command-routing speaker + whether to DROP the turn, from segments.

    Pure policy used by the live loops once :meth:`SpeakerPipeline.identify_segments`
    has named each diarized segment. Background TV / a son's chatter that the ECAPA
    threshold rejected is labelled ``None`` and contributes NO speech time, so it
    cannot drive a command:

    * Returns ``(speaker, False)`` where ``speaker`` is the enrolled person who spoke
      the MOST (by total segment duration) among the known (non-``None``) segments —
      conversation memory keys to them and the command is honoured.
    * Returns ``(None, True)`` (DROP) when EVERY segment is ``None`` (only unknown
      voices / TV) — there is no enrolled speaker to attribute the command to, so the
      loop must not let the chatter drive an LLM / music intent.
    * Returns ``(None, False)`` for an empty list (no audio) — guest, not a drop, so
      a typed/empty turn keeps its existing single-speaker semantics.
    """
    if not segments:
        return None, False
    durations: dict[str, float] = {}
    for start_s, end_s, name in segments:
        if name is not None:
            durations[name] = durations.get(name, 0.0) + max(0.0, end_s - start_s)
    if not durations:
        return None, True  # only unknown/TV -> drop the command
    winner = max(durations.items(), key=lambda kv: kv[1])[0]
    return winner, False
