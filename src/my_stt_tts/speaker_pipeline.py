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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .memory import GUEST_KEY
from .speaker_id import AMBIGUOUS, UNKNOWN, EcapaEmbedder, SpeakerIdentifier

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from .config import Config

log = logging.getLogger("my_stt_tts.speaker_pipeline")

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
    ) -> None:
        self.identifier = identifier
        self.embedder = embedder or EcapaEmbedder()
        self.sample_rate = sample_rate

    @classmethod
    def from_config(cls, cfg: Config) -> SpeakerPipeline | None:
        """Build the pipeline, or return ``None`` if speaker-ID is not usable.

        Gated and defensive: returns ``None`` (no model load, no latency) when
        disabled, when no enrolled centroids exist, or when ``speechbrain`` is not
        importable. Construction itself is wrapped so a broken install degrades to
        ``None`` instead of crashing the loop.
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
            return cls(identifier, sample_rate=getattr(cfg, "sample_rate", 16000))
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
        try:
            embedding = self.embedder.embed(clip, sample_rate=self.sample_rate)
            name = self.identifier.identify(embedding)
        except Exception as exc:  # noqa: BLE001 — degrade to guest, keep the turn alive
            log.warning("speaker ID failed for this utterance: %s", exc)
            return None
        if name in (UNKNOWN, AMBIGUOUS):
            return None
        return name
