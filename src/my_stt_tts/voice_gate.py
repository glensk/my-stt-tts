"""Only enrolled voices may start mac-voice: a speaker check on the wake-word audio.

Profiles are ECAPA centroids in ``enroll/<name>.npy`` (the same files ``scripts/enroll.py``
writes for the main pipeline's speaker ID). :func:`build_profile` makes one from the
wake-word clips a person recorded with ``scripts/enroll_wakeword.py -w <name>`` (saved as
``debug/recordings/wake/<word>/*-enroll_<name>-*.wav``), dropping outlier clips (silent,
clipped, someone else). :class:`VoiceGate` compares the audio around a wake fire with all
profiles; with no profiles at all it lets everyone through (and says so).

Threshold 0.35 (cosine) from Albert's data, 2026-10-09: his held-out clips 0.41–0.76,
five synthetic voices ≤ 0.23. Real family voices may sit closer than synthetic ones —
re-check with their recordings.
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# speechbrain/torch are heavy optional backends, imported lazily on purpose.
# pylint: disable=import-outside-toplevel

log = logging.getLogger("my_stt_tts.voice_gate")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = REPO_ROOT / "enroll"
WAKE_CLIPS = REPO_ROOT / "debug" / "recordings" / "wake"
THRESHOLD = 0.35
OUTLIER_COS = 0.2  # a clip this far from the others' mean is not the person (or no voice)


def _l2(vec: np.ndarray) -> np.ndarray:
    return vec / (np.linalg.norm(vec) + 1e-9)


def _read_wav(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path)) as w:
        raw = w.readframes(w.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


def default_embedder() -> Callable[[np.ndarray], np.ndarray]:
    """The ECAPA speaker-embedding function (16 kHz mono float32 -> 192-d vector)."""
    from .speaker_id import EcapaEmbedder

    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if (hub / "models--speechbrain--spkrec-ecapa-voxceleb").is_dir():
        # cached: load offline (no Hub round-trip, no "unauthenticated requests" warning)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = EcapaEmbedder()
    model.embed(np.zeros(16000, dtype=np.float32), 16000)  # load + warm up
    return lambda audio: np.asarray(model.embed(audio, 16000), dtype=np.float32)


def centroid(embeddings: list[np.ndarray]) -> tuple[np.ndarray, int]:
    """Mean of L2-normalised embeddings after dropping outliers; returns (centroid, kept)."""
    vecs = [_l2(np.asarray(e, dtype=np.float32)) for e in embeddings]
    keep = []
    for i, vec in enumerate(vecs):
        rest = [v for j, v in enumerate(vecs) if j != i]
        if not rest or float(vec @ _l2(np.mean(rest, axis=0))) >= OUTLIER_COS:
            keep.append(vec)
    return _l2(np.mean(keep or vecs, axis=0)), len(keep or vecs)


def clips_for(who: str, clips_dir: Path = WAKE_CLIPS) -> list[Path]:
    """Every wake-word clip recorded by ``who`` (any word)."""
    return sorted(clips_dir.glob(f"*/*-enroll_{who}-*.wav"))


def build_profile(
    who: str,
    *,
    embed: Callable[[np.ndarray], np.ndarray] | None = None,
    clips_dir: Path = WAKE_CLIPS,
    profile_dir: Path = PROFILE_DIR,
) -> tuple[Path | None, int, int]:
    """(Re)build ``<profile_dir>/<who>.npy`` from who's clips; (path|None, used, found)."""
    clips = clips_for(who, clips_dir)
    if len(clips) < 3:
        return None, 0, len(clips)
    embed = embed or default_embedder()
    vec, kept = centroid([embed(_read_wav(p)) for p in clips])
    profile_dir.mkdir(parents=True, exist_ok=True)
    path = profile_dir / f"{who}.npy"
    np.save(path, vec)
    return path, kept, len(clips)


def enrolled_speakers(clips_dir: Path = WAKE_CLIPS) -> list[str]:
    """Names that have tagged enrollment clips."""
    names = {
        p.name.split("-enroll_", 1)[1].rsplit("-", 1)[0]
        for p in clips_dir.glob("*/*.wav")
        if "-enroll_" in p.name
    }
    return sorted(names)


class VoiceGate:
    """Accept a wake fire only if its audio matches an enrolled profile."""

    def __init__(
        self,
        profile_dir: Path = PROFILE_DIR,
        *,
        threshold: float = THRESHOLD,
        embed: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.threshold = threshold
        self.profiles = {
            p.stem: _l2(np.load(p).astype(np.float32)) for p in sorted(profile_dir.glob("*.npy"))
        }
        self._embed = embed
        self._ready = threading.Event()
        if embed is not None or not self.profiles:
            self._ready.set()

    @property
    def active(self) -> bool:
        return bool(self.profiles)

    def preload(self) -> None:
        """Load the embedding model in the background (it takes ~5 s the first time)."""
        if self._ready.is_set():
            return

        def _load() -> None:
            try:
                self._embed = default_embedder()
            except Exception:  # pylint: disable=broad-exception-caught
                log.warning("⚠️  voice check unavailable; anyone can start", exc_info=True)
                self.profiles = {}
            self._ready.set()

        threading.Thread(target=_load, name="voice-gate-load", daemon=True).start()

    def check(self, audio: Any) -> tuple[bool, str, float]:
        """(accepted, best profile name, cosine). Everyone passes when nobody is enrolled."""
        if not self.profiles:
            return True, "", 0.0
        self._ready.wait(30)
        if self._embed is None:
            return True, "", 0.0
        clip = np.asarray(audio, dtype=np.float32).ravel()
        if clip.size < 4000:  # < 0.25 s: nothing to judge
            return False, "", 0.0
        emb = _l2(self._embed(clip))
        name, score = max(
            ((n, float(emb @ c)) for n, c in self.profiles.items()), key=lambda kv: kv[1]
        )
        return score >= self.threshold, name, score

    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        """Cosine of ``audio`` against ONE profile; None when it cannot be judged.

        Unlike :meth:`check` this never lets anyone through: no such profile, the model
        not loaded within ``timeout`` (or failed to load) or a clip under 0.25 s → None.
        """
        profile = self.profiles.get(name)
        if profile is None or not self._ready.wait(timeout) or self._embed is None:
            return None
        clip = np.asarray(audio, dtype=np.float32).ravel()
        if clip.size < 4000:
            return None
        return float(_l2(self._embed(clip)) @ profile)
