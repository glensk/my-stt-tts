"""Only enrolled voices may start mac-voice: a speaker check on the wake-word audio.

Profiles are ECAPA centroids in ``enroll/<name>.npy`` (the same files ``scripts/enroll.py``
writes for the main pipeline's speaker ID). :func:`build_profile` makes one from the
wake-word clips a person recorded with ``scripts/enroll_wakeword.py -w <name>`` (saved as
``debug/recordings/wake/<word>/*-enroll_<name>-*.wav``), dropping outlier clips (silent,
clipped, someone else). :class:`VoiceGate` compares the audio around a wake fire with all
profiles; with no profiles at all it lets everyone through (and says so). Once someone is
enrolled it fails CLOSED: when the speaker model fails to load (or is not loaded yet, or
crashes on a clip) no wake fire passes, so no call starts — one error line in the log and
one macOS banner say why (tp#1235).

Call-domain profiles in ``enroll/call/<name>.npy`` (``scripts/enroll_call.py``) are built
from sentences recorded through the same VoiceProcessingIO path a call uses. The wake gate
never sees them; :meth:`VoiceGate.score_against` (the bridge's in-call check) prefers one
over the wake profile, because a raw-mic wake profile scores in-call speech too low.

Negative clips for the bridge's margin rule (tp#1235) live in ``enroll/negatives/call.npy``
(one L2 embedding per row: in-call clips labelled ``other`` plus synthetic voices, written
by ``scripts/calibrate_voice.py -a``). :meth:`VoiceGate.score_detail` returns a clip's score
against a person's profile together with its best score against those negatives.

Threshold 0.35 (cosine) from Albert's data, 2026-10-09: his held-out clips 0.41–0.76,
five synthetic voices ≤ 0.23. Real family voices may sit closer than synthetic ones —
re-check with their recordings.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
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
CALL_SUBDIR = "call"  # enroll/call/<name>.npy: call-domain profiles (scripts/enroll_call.py)
NEGATIVES_FILE = Path("negatives") / "call.npy"  # under the profile dir (calibrate_voice.py)
WAKE_CLIPS = REPO_ROOT / "debug" / "recordings" / "wake"
THRESHOLD = 0.35
OUTLIER_COS = 0.2  # a clip this far from the others' mean is not the person (or no voice)
#: The one banner when the speaker model fails to load (no quotes: goes into AppleScript).
UNAVAILABLE_BANNER = "Voice check unavailable: wake word refused, no call starts. See the log."


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


def mac_banner(text: str) -> None:
    """Show ``text`` as a macOS notification titled mac-voice (never raises)."""
    script = f'display notification "{text}" with title "mac-voice"'
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(
            ["/usr/bin/osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )


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


def load_negatives(path: Path) -> np.ndarray | None:
    """The negative embeddings (rows, L2-normalised); None when there is no usable file."""
    try:
        rows = np.atleast_2d(np.load(path).astype(np.float32))
    except (OSError, ValueError):
        return None
    if rows.size == 0:
        return None
    return rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-9)


def _load_profiles(directory: Path) -> dict[str, np.ndarray]:
    """``{name: L2 centroid}`` for every ``*.npy`` directly in ``directory`` (none if absent)."""
    return {p.stem: _l2(np.load(p).astype(np.float32)) for p in sorted(directory.glob("*.npy"))}


class VoiceGate:  # pylint: disable=too-many-instance-attributes  # profiles + load state
    """Accept a wake fire only if its audio matches an enrolled profile (fail closed)."""

    def __init__(
        self,
        profile_dir: Path = PROFILE_DIR,
        *,
        threshold: float = THRESHOLD,
        embed: Callable[[np.ndarray], np.ndarray] | None = None,
        notify: Callable[[str], None] | None = None,
    ) -> None:
        self.threshold = threshold
        self.profiles = _load_profiles(profile_dir)  # wake gate (non-recursive: no call/)
        self.call_profiles = _load_profiles(profile_dir / CALL_SUBDIR)
        #: Negative clips for the margin rule (``enroll/negatives/call.npy``), or None.
        self.negatives = load_negatives(profile_dir / NEGATIVES_FILE)
        self._embed = embed
        self._notify = notify or mac_banner
        #: True once the speaker model failed to load: every wake fire is refused.
        self.load_failed = False
        self._notified = False
        self._ready = threading.Event()
        if embed is not None or not (self.profiles or self.call_profiles):
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
                log.exception(
                    "❌ voice check unavailable: the speaker model failed to load — the wake"
                    " word is refused and no call starts until mac-voice is restarted"
                )
                self._load_failed()
            self._ready.set()

        threading.Thread(target=_load, name="voice-gate-load", daemon=True).start()

    def _load_failed(self) -> None:
        """Fail closed from now on, and show the one banner that says so."""
        self.load_failed = True
        if not self._notified:
            self._notified = True
            self._notify(UNAVAILABLE_BANNER)

    def unavailable(self) -> str | None:
        """Why :meth:`check` refuses everyone right now (the model), or None."""
        if not self.profiles:
            return None
        if self.load_failed:
            return "speaker model failed to load"
        if self._embed is None:
            return "speaker model not loaded yet"
        return None

    def check(self, audio: Any) -> tuple[bool, str, float]:
        """(accepted, best profile name, cosine). Everyone passes when nobody is enrolled.

        With profiles it fails closed: the model failed to load, is not loaded within 30 s
        or raises on the clip → refused (:meth:`unavailable` says why for the first two).
        """
        if not self.profiles:
            return True, "", 0.0
        if not self._ready.wait(30) or self._embed is None:
            return False, "", 0.0
        clip = np.asarray(audio, dtype=np.float32).ravel()
        if clip.size < 4000:  # < 0.25 s: nothing to judge
            return False, "", 0.0
        try:
            emb = _l2(self._embed(clip))
        except Exception:  # pylint: disable=broad-exception-caught  # fail closed
            log.warning("⚠️  voice check failed on a wake clip — refused", exc_info=True)
            return False, "", 0.0
        name, score = max(
            ((n, float(emb @ c)) for n, c in self.profiles.items()), key=lambda kv: kv[1]
        )
        return score >= self.threshold, name, score

    def profile_kind(self, name: str) -> str | None:
        """Which profile :meth:`score_against` uses for ``name``: ``call``, ``wake`` or None."""
        if name in self.call_profiles:
            return "call"
        return "wake" if name in self.profiles else None

    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        """Cosine of ``audio`` against ONE person's profile; None when it cannot be judged.

        Uses the call-domain profile for ``name`` when there is one, else the wake profile
        (:meth:`profile_kind` says which). Unlike :meth:`check` this never lets anyone
        through: no such profile, the model not loaded within ``timeout`` (or failed to
        load) or a clip under 0.25 s → None.
        """
        profile = self.call_profiles.get(name)
        if profile is None:
            profile = self.profiles.get(name)
        if profile is None or not self._ready.wait(timeout) or self._embed is None:
            return None
        clip = np.asarray(audio, dtype=np.float32).ravel()
        if clip.size < 4000:
            return None
        return float(_l2(self._embed(clip)) @ profile)

    def score_detail(
        self, audio: Any, name: str, *, timeout: float = 5.0
    ) -> tuple[float, float | None] | None:
        """(score against ``name``'s profile, best score against the negatives or None).

        The same rules as :meth:`score_against` (None when it cannot be judged); one
        embedding serves both numbers. The second is None when there are no negatives.
        """
        profile = self.call_profiles.get(name)
        if profile is None:
            profile = self.profiles.get(name)
        if profile is None or not self._ready.wait(timeout) or self._embed is None:
            return None
        clip = np.asarray(audio, dtype=np.float32).ravel()
        if clip.size < 4000:
            return None
        emb = _l2(self._embed(clip))
        negatives = self.negatives
        best = None if negatives is None else float(np.max(negatives @ emb))
        return float(emb @ profile), best
