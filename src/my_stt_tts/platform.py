"""Platform detection + cross-platform playback selection (G8).

The pipeline was built macOS-first (parakeet-mlx STT, ``afplay`` playback, macOS
VoiceProcessingIO AEC). To run the central "brain" on a **Linux** box with Mac /
ESP32 satellites, the OS-specific seams need a Linux path. This module:

* detects the host OS (``detect_platform``), honouring an explicit ``cfg.platform``
  override so tests / cross-host setups can pin it;
* selects a **playback** command for rendered WAVs (``afplay`` on macOS;
  ``aplay`` / ``paplay`` on Linux) via :func:`select_player`, used by the
  cancellable :class:`~my_stt_tts.tts.Playback` path;
* plays a numpy PCM array through the best available sink (:func:`play_array`):
  ``sounddevice`` when present (cross-platform), else a subprocess player on Linux.

The macOS path is **unchanged** when the platform auto-detects to macOS — the
existing ``afplay`` / sounddevice behaviour is preserved. Everything here is pure
selection logic (string/command choice) so it is unit-tested with fakes — no
device, no subprocess in tests.
"""

from __future__ import annotations

import logging
import shutil
import sys
from typing import Any

log = logging.getLogger("my_stt_tts.platform")

MACOS = "macos"
LINUX = "linux"
OTHER = "other"


def detect_platform(cfg: Any = None) -> str:
    """Return ``"macos"`` / ``"linux"`` / ``"other"``, honouring ``cfg.platform``.

    An explicit ``cfg.platform`` of ``"macos"`` / ``"linux"`` overrides the
    auto-detection (handy for tests and for documenting intent); ``"auto"`` (the
    default) falls through to :data:`sys.platform`.
    """
    override = getattr(cfg, "platform", "auto") if cfg is not None else "auto"
    if override in (MACOS, LINUX):
        return override
    if sys.platform == "darwin":
        return MACOS
    if sys.platform.startswith("linux"):
        return LINUX
    return OTHER


def is_macos(cfg: Any = None) -> bool:
    """True when the resolved platform is macOS."""
    return detect_platform(cfg) == MACOS


# Candidate WAV players per platform, in preference order. The first one found on
# PATH is used. ``afplay`` is macOS-only; ``aplay`` (ALSA) / ``paplay`` (PulseAudio)
# are the common Linux CLI players; ``ffplay`` is a cross-platform last resort.
_PLAYERS: dict[str, tuple[str, ...]] = {
    MACOS: ("afplay", "ffplay"),
    LINUX: ("aplay", "paplay", "ffplay"),
    OTHER: ("ffplay", "aplay", "afplay"),
}


def select_player(cfg: Any = None, *, which: Any = shutil.which) -> tuple[str, ...] | None:
    """Return the argv prefix for a WAV-file player, or None if none is available.

    ``cfg.playback_backend`` can pin a specific player (``afplay`` / ``aplay``);
    otherwise the per-platform candidate list is probed against PATH. Returns an
    argv *prefix* (the WAV path is appended by the caller), e.g. ``("aplay", "-q")``.
    ``which`` is injectable so the probe is testable without real binaries.
    """
    pinned = getattr(cfg, "playback_backend", "auto") if cfg is not None else "auto"
    plat = detect_platform(cfg)
    if pinned not in ("auto", "sounddevice") and which(pinned):
        return _player_argv(pinned)
    for name in _PLAYERS.get(plat, _PLAYERS[OTHER]):
        if which(name):
            return _player_argv(name)
    return None


def _player_argv(name: str) -> tuple[str, ...]:
    """Map a player name to its quiet-mode argv prefix (the WAV path is appended)."""
    if name == "aplay":
        return ("aplay", "-q")
    if name == "paplay":
        return ("paplay",)
    if name == "ffplay":
        return ("ffplay", "-autoexit", "-nodisp", "-loglevel", "quiet")
    return (name,)  # afplay <path>


def play_array(samples: Any, sample_rate: int, cfg: Any = None) -> None:
    """Play a float32 mono PCM array on the best available sink (cross-platform).

    Prefers ``sounddevice`` (works on macOS + Linux + Windows when the ``audio``
    extra is installed); on a Linux box without it, falls back to writing a temp
    WAV and invoking the selected CLI player (``aplay`` / ``paplay``). Raises only
    if neither path is available, so a headless host fails loudly rather than
    silently dropping audio.
    """
    if getattr(cfg, "playback_backend", "auto") not in ("aplay", "paplay", "afplay"):
        try:
            from . import audio

            sd = audio._sd()  # noqa: SLF001 — same-package lazy accessor
            sd.play(samples, samplerate=sample_rate)
            sd.wait()
            return
        except Exception:  # sounddevice/PortAudio unavailable -> CLI fallback
            log.info("sounddevice unavailable; using a CLI player.", exc_info=True)
    _play_array_via_cli(samples, sample_rate, cfg)


def _play_array_via_cli(samples: Any, sample_rate: int, cfg: Any) -> None:
    """Write a temp WAV and play it via the selected CLI player (Linux fallback)."""
    import subprocess
    import tempfile
    from pathlib import Path

    from .util import wav_bytes_from_float

    player = select_player(cfg)
    if player is None:
        raise RuntimeError("no audio playback backend available (no sounddevice, no CLI player)")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        out = handle.name
        handle.write(wav_bytes_from_float(samples, sample_rate))
    try:
        subprocess.run([*player, out], check=False)  # noqa: S603
    finally:
        Path(out).unlink(missing_ok=True)
