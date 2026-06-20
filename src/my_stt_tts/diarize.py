"""Within-turn speaker diarization via sherpa-onnx (G7+).

Splits ONE captured turn (a 16 kHz mono float32 clip) into anonymous per-speaker
segments ``[(start_s, end_s, local_speaker_id)]`` using sherpa-onnx's **offline**
speaker diarization (its pyannote segmentation model + a speaker-embedding extractor
+ fast clustering). This module ONLY produces the *anonymous* segmentation — naming
each segment with an enrolled person reuses the existing SpeechBrain ECAPA path in
:mod:`my_stt_tts.speaker_pipeline`, so ``enroll/*.npy`` centroids +
``scripts/calibrate.py`` keep working unchanged.

Everything here is **gated + fully defensive**, mirroring :mod:`speaker_pipeline`:

* sherpa-onnx (the ``diarize`` extra) is lazy-imported — the core package and every
  other extra import without it;
* the segmentation + embedding ONNX models are auto-downloaded ONCE into the
  gitignored ``models/`` and **checksum-verified** (a truncated/tampered download is
  rejected, never used);
* if sherpa or the models are unavailable, or diarization raises on a clip,
  :meth:`SherpaDiarizer.segments` returns a **single whole-clip segment** so the
  caller transparently degrades to today's single-speaker behaviour. A turn must
  NEVER die because diarization hiccuped.

The sherpa embedding model is used solely for sherpa's internal clustering (grouping
frames into anonymous local speakers); it is NOT the model that names a segment.
"""

from __future__ import annotations

import bz2
import contextlib
import logging
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .turn import verify_checksum

if TYPE_CHECKING:  # avoid a hard import cycle at module load
    from .config import Config

log = logging.getLogger("my_stt_tts.diarize")

# A diarized segment: (start_seconds, end_seconds, anonymous_local_speaker_id).
Segment = tuple[float, float, int]


def _sherpa_importable() -> bool:
    """True if ``import sherpa_onnx`` actually succeeds (not just that it's installed).

    ``importlib.util.find_spec`` only proves the package directory exists; the native
    C-extension can still fail to load. We probe the real import so the gating in
    :meth:`SherpaDiarizer.from_config` skips an unusable install rather than downloading
    models for it. Any import-time error reads as "not usable".

    The ``diarize`` extra pins ``sherpa-onnx==1.10.46`` precisely because it is the last
    macOS arm64 wheel that *self-bundles* its onnxruntime (``libonnxruntime.1.17.1.dylib``
    inside ``sherpa_onnx/lib/``). Newer (1.12.26+) wheels dropped the bundled dylib yet
    still hard-link ``@rpath/libonnxruntime.1.24.4.dylib``, which the standalone
    ``onnxruntime`` pip package does NOT satisfy (its dylib lives in ``onnxruntime/capi/``,
    off sherpa's rpath) — so they ``dlopen``-fail. This probe is the runtime guard against
    re-introducing such a broken pairing: it degrades to single-speaker instead of crashing.
    """
    try:
        import sherpa_onnx  # noqa: F401 — import probe only
    except Exception:  # noqa: BLE001 — any load failure means sherpa is unusable here
        return False
    return True


def whole_clip_segment(clip: np.ndarray, sample_rate: int) -> list[Segment]:
    """A single segment spanning the entire clip (the single-speaker fallback).

    Returned whenever diarization is off/unavailable/failed so the caller treats the
    turn as one anonymous speaker — i.e. exactly today's behaviour. Empty clip -> [].
    """
    n = int(np.asarray(clip).size)
    if n == 0:
        return []
    return [(0.0, n / float(sample_rate), 0)]


def _http_get(url: str, timeout: int = 120) -> bytes:
    """Fetch ``url`` and return its bytes (pinned HTTPS release URLs only)."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — pinned HTTPS
        data: bytes = resp.read()
    return data


def _extract_segmentation_onnx(archive_bytes: bytes, dest: Path) -> bool:
    """Extract the pyannote ``model.onnx`` from the sherpa segmentation .tar.bz2.

    The release ships a ``.tar.bz2`` whose ``<dir>/model.onnx`` is the float32
    segmentation model. We unpack to a temp dir, find that member, and atomically
    move it to ``dest``. Returns whether ``dest`` now exists. Defensive: a malformed
    archive / missing member / IO error is logged and reported as a failure.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = bz2.decompress(archive_bytes)
            tar_path = Path(tmpdir) / "seg.tar"
            tar_path.write_bytes(raw)
            with tarfile.open(tar_path) as tar:
                member = next(
                    (
                        m
                        for m in tar.getmembers()
                        if m.isfile() and Path(m.name).name == "model.onnx"
                    ),
                    None,
                )
                if member is None:
                    log.warning("segmentation archive has no model.onnx member")
                    return False
                extracted = tar.extractfile(member)
                if extracted is None:
                    return False
                payload = extracted.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(payload)
            tmp.replace(dest)
    except (OSError, tarfile.TarError, EOFError, ValueError):
        log.warning("failed to extract the segmentation model", exc_info=True)
        with contextlib.suppress(OSError):
            (dest.with_suffix(dest.suffix + ".part")).unlink(missing_ok=True)
        return False
    return dest.is_file()


def ensure_segmentation_model(
    model_path: str,
    url: str,
    *,
    auto_download: bool = True,
    expected_sha256: str = "",
) -> bool:
    """Ensure the pyannote segmentation ``model.onnx`` exists, downloading if needed.

    Mirrors :func:`my_stt_tts.turn.ensure_smart_turn_model` but unpacks the sherpa
    ``.tar.bz2`` first. A present-but-corrupt file (failing ``expected_sha256``) is
    dropped and re-fetched; a download whose extracted model fails the checksum is
    discarded rather than installed. Network/IO failures are swallowed (the caller
    then degrades to the single-speaker fallback). Returns whether the file is ready.
    """
    path = Path(model_path)
    if path.is_file():
        if verify_checksum(path, expected_sha256):
            return True
        log.warning("segmentation model at %s failed checksum; re-downloading.", model_path)
        with contextlib.suppress(OSError):
            path.unlink()
    if not auto_download or not url:
        return False
    log.info("downloading diarization segmentation model %s ...", url)
    try:
        archive = _http_get(url)
    except (urllib.error.URLError, OSError, ValueError):
        log.warning("segmentation model download failed.", exc_info=True)
        return False
    if not archive or not _extract_segmentation_onnx(archive, path):
        return False
    if not verify_checksum(path, expected_sha256):
        log.warning("segmentation model checksum mismatch after extract; discarding.")
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)
        return False
    return path.is_file()


def ensure_embedding_model(
    model_path: str,
    url: str,
    *,
    auto_download: bool = True,
    expected_sha256: str = "",
) -> bool:
    """Ensure the sherpa speaker-embedding ``.onnx`` exists, downloading if needed.

    The embedding model is a bare ``.onnx`` (no archive). Same checksum-verified,
    atomic, failure-swallowing semantics as :func:`ensure_segmentation_model`.
    """
    path = Path(model_path)
    if path.is_file():
        if verify_checksum(path, expected_sha256):
            return True
        log.warning("embedding model at %s failed checksum; re-downloading.", model_path)
        with contextlib.suppress(OSError):
            path.unlink()
    if not auto_download or not url:
        return False
    log.info("downloading diarization embedding model %s ...", url)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        data = _http_get(url)
        if not data:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(data)
        if not verify_checksum(tmp, expected_sha256):
            log.warning("embedding model checksum mismatch; discarding.")
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
            return False
        tmp.replace(path)
    except (urllib.error.URLError, OSError, ValueError):
        log.warning("embedding model download failed.", exc_info=True)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        return False
    return path.is_file()


class SherpaDiarizer:
    """Lazy sherpa-onnx offline speaker diarizer (anonymous per-speaker segments).

    Loads nothing until the first :meth:`segments` call (and caches the built engine).
    :meth:`segments` returns ``[(start_s, end_s, local_speaker_id)]`` sorted by start
    time, or the single whole-clip fallback when sherpa is unavailable / a clip is
    too short / inference raises — so the loop transparently keeps working.
    """

    # sherpa diarization works on 16 kHz mono; below ~1 s a turn is one speaker.
    _MIN_SAMPLES = 16000  # 1.0 s @ 16 kHz

    def __init__(
        self,
        segmentation_model_path: str,
        embedding_model_path: str,
        *,
        num_speakers: int = -1,
        cluster_threshold: float = 0.5,
        min_segment_s: float = 0.4,
        sample_rate: int = 16000,
    ) -> None:
        self.segmentation_model_path = segmentation_model_path
        self.embedding_model_path = embedding_model_path
        self.num_speakers = num_speakers
        self.cluster_threshold = cluster_threshold
        self.min_segment_s = min_segment_s
        self.sample_rate = sample_rate
        self._engine: Any = None
        self._unavailable = False  # latched once we know sherpa/models can't load

    @classmethod
    def from_config(cls, cfg: Config) -> SherpaDiarizer | None:
        """Build a diarizer, or ``None`` if diarization is not usable.

        Gated + defensive: returns ``None`` when diarization is disabled, when
        sherpa-onnx is not importable, or when the models cannot be made available
        (download off + absent, or a download/checksum failure). Construction itself
        is wrapped so a broken install degrades to ``None`` (single-speaker) instead
        of crashing the loop. The heavy ONNX engine is still built lazily on first use.
        """
        if not getattr(cfg, "speaker_diarize_enabled", False):
            return None
        try:
            import importlib.util

            if importlib.util.find_spec("sherpa_onnx") is None:
                log.warning("diarization enabled but `sherpa-onnx` is not installed; skipping")
                return None
            # find_spec only proves the package dir exists; the C-extension can still
            # fail to dlopen (e.g. an onnxruntime ABI mismatch). Probe a real import
            # BEFORE downloading ~45 MB of models for an unusable install.
            if not _sherpa_importable():
                log.warning("diarization enabled but `sherpa-onnx` failed to import; skipping")
                return None
            seg_ok = ensure_segmentation_model(
                cfg.diarize_segmentation_model_path,
                cfg.diarize_segmentation_url,
                auto_download=getattr(cfg, "diarize_auto_download", True),
                expected_sha256=getattr(cfg, "diarize_segmentation_sha256", ""),
            )
            emb_ok = ensure_embedding_model(
                cfg.diarize_embedding_model_path,
                cfg.diarize_embedding_url,
                auto_download=getattr(cfg, "diarize_auto_download", True),
                expected_sha256=getattr(cfg, "diarize_embedding_sha256", ""),
            )
            if not (seg_ok and emb_ok):
                log.warning("diarization enabled but models unavailable; single-speaker fallback")
                return None
            log.info("within-turn diarization active (sherpa-onnx)")
            return cls(
                cfg.diarize_segmentation_model_path,
                cfg.diarize_embedding_model_path,
                num_speakers=getattr(cfg, "diarize_num_speakers", -1),
                cluster_threshold=getattr(cfg, "diarize_cluster_threshold", 0.5),
                min_segment_s=getattr(cfg, "diarize_min_segment_s", 0.4),
                sample_rate=getattr(cfg, "sample_rate", 16000),
            )
        except Exception as exc:  # noqa: BLE001 — never let diarization setup break the loop
            log.warning("diarization disabled (setup failed): %s", exc)
            return None

    def _ensure_engine(self) -> Any:
        """Build (once) and return the sherpa OfflineSpeakerDiarization engine, or None."""
        if self._engine is not None or self._unavailable:
            return self._engine
        try:
            import sherpa_onnx

            config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
                segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                    pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                        model=self.segmentation_model_path
                    ),
                ),
                embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                    model=self.embedding_model_path
                ),
                clustering=sherpa_onnx.FastClusteringConfig(
                    num_clusters=self.num_speakers,
                    threshold=self.cluster_threshold,
                ),
                min_duration_on=0.3,
                min_duration_off=0.5,
            )
            if not config.validate():
                log.warning("sherpa diarization config invalid; single-speaker fallback")
                self._unavailable = True
                return None
            self._engine = sherpa_onnx.OfflineSpeakerDiarization(config)
        except Exception as exc:  # noqa: BLE001 — degrade to single-speaker, keep the turn alive
            log.warning("sherpa diarization engine load failed: %s", exc)
            self._unavailable = True
            return None
        return self._engine

    def segments(self, clip: np.ndarray | None) -> list[Segment]:
        """Diarize ``clip`` into ``[(start_s, end_s, local_speaker_id)]`` (sorted).

        Returns the single whole-clip segment when diarization is unavailable, the
        clip is too short to diarize, no segments are found, or inference raises — so
        a turn never dies and the caller transparently degrades to single-speaker.
        Segments shorter than ``min_segment_s`` are dropped (noise / clipped frames).
        """
        if clip is None or not isinstance(clip, np.ndarray) or clip.size == 0:
            return []
        audio = np.ascontiguousarray(np.asarray(clip, dtype=np.float32).ravel())
        if audio.size < self._MIN_SAMPLES:
            return whole_clip_segment(audio, self.sample_rate)
        engine = self._ensure_engine()
        if engine is None:
            return whole_clip_segment(audio, self.sample_rate)
        try:
            result = engine.process(audio).sort_by_start_time()
            segs: list[Segment] = [(float(r.start), float(r.end), int(r.speaker)) for r in result]
        except Exception as exc:  # noqa: BLE001 — degrade to single-speaker, keep the turn alive
            log.warning("diarization failed for this turn: %s", exc)
            return whole_clip_segment(audio, self.sample_rate)
        segs = [s for s in segs if (s[1] - s[0]) >= self.min_segment_s]
        if not segs:
            return whole_clip_segment(audio, self.sample_rate)
        return segs
