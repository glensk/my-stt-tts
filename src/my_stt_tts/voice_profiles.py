"""Voice data on disk: enrollment takes, profile rebuilds and private files (tp#1235).

``scripts/enroll_voice.py`` keeps every take it records under ``enroll/<who>/<stamp>/``:

* ``session.json`` — who, when, how it was recorded (``vpio``: through VoiceProcessingIO like
  a call, ``raw``: plain mic) and one entry per take (sentence, language, ``short`` / ``long``,
  file names, seconds of speech, whether it was usable);
* ``call/NN-aK.wav`` — the take as a call hears it (VoiceProcessingIO);
* ``raw/NN-aK.wav`` — the same take from the plain mic (the wake word's domain), when that
  second stream could be opened, or the only copy when VoiceProcessingIO was unavailable.

:func:`rebuild_profiles` rebuilds BOTH profiles from every usable take of every session:
the call profile ``enroll/call/<who>.npy`` (what the bridge's ``Authoriser`` checks) from the
``call/`` takes, and the wake profile ``enroll/<who>.npy`` (who may start a call) from the
wake-word clips plus the ``raw/`` takes. The previous profiles are copied to
``enroll/<who>/backup/`` first, and ``enroll/<who>/profiles.json`` records which takes built
which profile (and its SHA-256), so ``scripts/calibrate_voice.py`` can score the takes
leave-one-out against exactly that profile.

All of it is personal voice data: files are written 0600, folders 0700, and ``enroll/`` and
``debug/`` are gitignored.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import logging
import os
import shutil
import tempfile
import time
import wave
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .turns import TurnSource
from .voice_gate import CALL_SUBDIR, PROFILE_DIR, REPO_ROOT, WAKE_CLIPS, centroid, clips_for

log = logging.getLogger("my_stt_tts.voice_profiles")

SAMPLE_RATE = 16000
FRAME_SAMPLES = 4000  # eleven_voice.INPUT_CHUNK: the frames the bridge's VAD sees in a call
PRIVATE_FILE = 0o600
PRIVATE_DIR = 0o700
NEGATIVES_SUBDIR = "negatives"  # enroll/negatives/: never a person's profile (*.npy below it)
#: A take needs this much speech to count (shorter ones are kept but not used).
MIN_TAKE_SPEECH_S = 1.0
#: At least this many usable takes before a profile is (re)built.
MIN_PROFILE_TAKES = 3
ENV_FILE = REPO_ROOT / ".env"


# -- private files ---------------------------------------------------------------------------
def private_dir(path: Path) -> Path:
    """``path`` as a folder only its owner can read (created with its parents)."""
    path.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(path, PRIVATE_DIR)
    return path


def _private_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically with mode 0600 (never readable by others)."""
    private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=path.parent)
    try:
        os.fchmod(fd, PRIVATE_FILE)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def wav_bytes(pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """16-bit mono WAV file contents for float32 ``pcm`` in [-1, 1]."""
    samples = (np.clip(np.asarray(pcm, dtype=np.float32).ravel(), -1.0, 1.0) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.Wave_write(buf) as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(samples.tobytes())
    return buf.getvalue()


def write_wav(path: Path, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> Path:
    """Save ``pcm`` as a private 16-bit mono WAV."""
    _private_write(path, wav_bytes(pcm, sample_rate))
    return path


def write_json(path: Path, data: Any) -> Path:
    """Save ``data`` as private, indented JSON."""
    _private_write(path, (json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode())
    return path


def read_json(path: Path) -> Any:
    """The JSON in ``path``; None when it is missing or broken."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Mono float32 PCM at ``sample_rate`` (other rates are resampled linearly)."""
    with wave.open(str(path)) as w:
        rate, channels, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError(f"{path}: only 16-bit WAV files are supported")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    if rate != sample_rate and pcm.size:
        n_out = round(pcm.size * sample_rate / rate)
        pcm = np.interp(np.linspace(0, pcm.size - 1, n_out), np.arange(pcm.size), pcm).astype(
            np.float32
        )
    return pcm


def sha256_file(path: Path) -> str | None:
    """Hex SHA-256 of ``path``; None when it does not exist."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def save_profile(path: Path, vec: np.ndarray, backup: Path | None = None) -> Path | None:
    """Write a profile ``.npy`` (0600); first copy the old one to ``backup``. Returns the copy."""
    copied = None
    if backup is not None and path.is_file():
        private_dir(backup.parent)
        shutil.copyfile(path, backup)
        os.chmod(backup, PRIVATE_FILE)
        copied = backup
    buf = io.BytesIO()
    np.save(buf, np.asarray(vec, dtype=np.float32))
    _private_write(path, buf.getvalue())
    return copied


# -- vectors ---------------------------------------------------------------------------------
def l2(vec: Any) -> np.ndarray:
    """``vec`` scaled to unit length (float32, flat)."""
    arr = np.asarray(vec, dtype=np.float32).ravel()
    return arr / (float(np.linalg.norm(arr)) + 1e-9)


def mean_profile(embeddings: list[np.ndarray]) -> np.ndarray:
    """L2-normalised mean of L2-normalised embeddings."""
    return l2(np.mean([l2(e) for e in embeddings], axis=0))


def leave_one_out(embeddings: list[np.ndarray]) -> list[float]:
    """Per embedding: cosine against the mean profile of all the OTHER ones."""
    if len(embeddings) < 2:
        return [float("nan")] * len(embeddings)
    return [
        float(l2(e) @ mean_profile(embeddings[:i] + embeddings[i + 1 :]))
        for i, e in enumerate(embeddings)
    ]


def best_match(emb: np.ndarray, others: np.ndarray | None) -> float | None:
    """Highest cosine of ``emb`` against the rows of ``others``; None when there are none."""
    if others is None or np.asarray(others).size == 0:
        return None
    return float(np.max(np.asarray(others, dtype=np.float32) @ l2(emb)))


# -- speech trimming -------------------------------------------------------------------------
def speech_only(clip: np.ndarray, vad: Any) -> tuple[np.ndarray, float]:
    """(speech PCM, seconds): the utterances the bridge's segmentation finds in ``clip``."""
    turns = TurnSource(vad, sample_rate=SAMPLE_RATE, ring_s=1e9)
    at = 0.0
    clip = np.asarray(clip, dtype=np.float32).ravel()
    padded = np.concatenate([clip, np.zeros(3 * FRAME_SAMPLES, dtype=np.float32)])
    for start in range(0, padded.size, FRAME_SAMPLES):
        frame = padded[start : start + FRAME_SAMPLES]
        at += frame.size / SAMPLE_RATE
        turns.feed(frame, at=at)
    utts = turns.utterances()
    if not utts:
        return np.zeros(0, dtype=np.float32), 0.0
    return np.concatenate([u.pcm for u in utts]), sum(u.duration for u in utts)


# -- enrollment sessions ---------------------------------------------------------------------
@dataclass(frozen=True)
class Take:
    """One kept recording of a sentence (``domain``: ``call`` = VoiceProcessingIO, ``raw``)."""

    path: Path
    domain: str
    lang: str = ""
    kind: str = ""  # "short" (a command) / "long"
    session: str = ""


def person_dir(who: str, root: Path = PROFILE_DIR) -> Path:
    """``enroll/<who>/``: that person's kept takes, backups and ``profiles.json``."""
    return root / who


def session_dirs(who: str, root: Path = PROFILE_DIR) -> list[Path]:
    """Every enrollment session folder of ``who`` (oldest first)."""
    base = person_dir(who, root)
    return sorted(p.parent for p in base.glob("*/session.json"))


def usable_takes(who: str, root: Path = PROFILE_DIR) -> list[Take]:
    """Every usable take of every session (both domains), oldest first."""
    takes: list[Take] = []
    for folder in session_dirs(who, root):
        data = read_json(folder / "session.json") or {}
        for entry in data.get("takes", []):
            if not entry.get("usable"):
                continue
            for domain, key in (("call", "call_file"), ("raw", "raw_file")):
                name = entry.get(key)
                if name and (folder / name).is_file():
                    takes.append(
                        Take(
                            folder / name,
                            domain,
                            str(entry.get("lang", "")),
                            str(entry.get("kind", "")),
                            folder.name,
                        )
                    )
    return takes


@dataclass
class Rebuild:  # pylint: disable=too-many-instance-attributes  # one report
    """What :func:`rebuild_profiles` did."""

    call_path: Path | None = None
    call_domain: str = ""  # "call" (VoiceProcessingIO takes) or "raw" (fallback)
    call_used: int = 0
    wake_path: Path | None = None
    wake_used: int = 0
    backups: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    #: Speech embedding per take path (what the profiles were built from).
    embeddings: dict[Path, np.ndarray] = field(default_factory=dict)


def embed_takes(
    paths: Iterable[Path],
    embed: Callable[[np.ndarray], np.ndarray],
    vad: Any,
    *,
    min_speech_s: float = MIN_TAKE_SPEECH_S,
) -> dict[Path, np.ndarray]:
    """Speech embedding per recording (VAD-trimmed); recordings with too little speech skip."""
    out: dict[Path, np.ndarray] = {}
    for path in paths:
        speech, seconds = speech_only(read_wav(path), vad)
        if seconds >= min_speech_s:
            out[path] = l2(embed(speech))
    return out


def rebuild_profiles(  # pylint: disable=too-many-arguments,too-many-locals
    who: str,
    embed: Callable[[np.ndarray], np.ndarray],
    vad: Any,
    *,
    root: Path = PROFILE_DIR,
    wake_clips: Path = WAKE_CLIPS,
    stamp: str | None = None,
    write: bool = True,
) -> Rebuild:
    """Rebuild the call and wake profiles of ``who`` from ALL kept takes (see module doc).

    Wake-word clips are short and already cut, so they are embedded whole (as
    ``voice_gate.build_profile`` does); takes are VAD-trimmed first. ``write=False``
    computes everything and writes nothing.
    """
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    result = Rebuild()
    takes = usable_takes(who, root)
    call_takes = [t.path for t in takes if t.domain == "call"]
    raw_takes = [t.path for t in takes if t.domain == "raw"]
    result.embeddings = embed_takes(call_takes + raw_takes, embed, vad)
    call_src = [p for p in call_takes if p in result.embeddings]
    result.call_domain = "call"
    if not call_src:
        call_src = [p for p in raw_takes if p in result.embeddings]
        result.call_domain = "raw"
        if call_src:
            result.notes.append(
                "no VoiceProcessingIO takes: the call profile is built from plain-mic takes,"
                " which score in-call speech lower — re-enroll with VoiceProcessingIO when it works"
            )
    wake_embs = {p: l2(embed(read_wav(p))) for p in clips_for(who, wake_clips)}
    wake_src = list(wake_embs) + [p for p in raw_takes if p in result.embeddings]
    if not any(p in result.embeddings for p in raw_takes):
        result.notes.append(
            "no plain-mic takes: the wake profile is rebuilt from the wake-word clips only"
        )
    manifest: dict[str, Any] = {"who": who, "built": stamp}
    person = person_dir(who, root)
    if len(call_src) >= MIN_PROFILE_TAKES:
        vec, kept = centroid([result.embeddings[p] for p in call_src])
        target = root / CALL_SUBDIR / f"{who}.npy"
        if write:
            backup = save_profile(target, vec, person / "backup" / f"{stamp}-call.npy")
            result.backups += [backup] if backup else []
        result.call_path, result.call_used = target, kept
        manifest["call"] = {
            "file": str(target.relative_to(root)),
            "domain": result.call_domain,
            "sha256": sha256_file(target) if write else None,
            "takes": [str(p.relative_to(root)) for p in call_src],
        }
    else:
        result.notes.append(f"only {len(call_src)} usable take(s): call profile unchanged")
    if len(wake_src) >= MIN_PROFILE_TAKES:
        all_embs = {**wake_embs, **result.embeddings}
        vec, kept = centroid([all_embs[p] for p in wake_src])
        target = root / f"{who}.npy"
        if write:
            backup = save_profile(target, vec, person / "backup" / f"{stamp}-wake.npy")
            result.backups += [backup] if backup else []
        result.wake_path, result.wake_used = target, kept
        manifest["wake"] = {
            "file": str(target.relative_to(root)),
            "sha256": sha256_file(target) if write else None,
            "sources": [_rel(p, root) for p in wake_src],
        }
    if write and (result.call_path or result.wake_path):
        write_json(person / "profiles.json", manifest)
    return result


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def load_profile(path: Path) -> np.ndarray | None:
    """An L2 profile vector, or None when there is no such file."""
    try:
        return l2(np.load(path))
    except (OSError, ValueError):
        return None


# -- .env values -----------------------------------------------------------------------------
def env_value(key: str, env_file: Path = ENV_FILE, env: Mapping[str, str] | None = None) -> str:
    """``key`` from the process environment, else from the repo ``.env`` (only that key)."""
    env = os.environ if env is None else env
    if env.get(key, "").strip():
        return env[key].strip()
    from dotenv import dotenv_values  # pylint: disable=import-outside-toplevel

    value = dotenv_values(env_file).get(key) if env_file.is_file() else None
    return (value or "").strip()


def env_float(
    key: str,
    default: float,
    env_file: Path = ENV_FILE,
    env: Mapping[str, str] | None = None,
) -> float:
    """``key`` as a non-negative float (environment, then ``.env``); ``default`` otherwise."""
    raw = env_value(key, env_file, env)
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if np.isfinite(value) and value >= 0 else default


__all__ = [
    "MIN_PROFILE_TAKES",
    "MIN_TAKE_SPEECH_S",
    "NEGATIVES_SUBDIR",
    "WAKE_CLIPS",
    "Rebuild",
    "Take",
    "best_match",
    "embed_takes",
    "env_float",
    "env_value",
    "leave_one_out",
    "load_profile",
    "mean_profile",
    "private_dir",
    "read_json",
    "read_wav",
    "rebuild_profiles",
    "save_profile",
    "session_dirs",
    "sha256_file",
    "speech_only",
    "usable_takes",
    "write_json",
    "write_wav",
]
