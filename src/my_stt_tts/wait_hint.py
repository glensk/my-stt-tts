"""The spoken "One moment." while a mac-voice client tool is slow (tp#1234).

A decide read on a phone hotspot can take seconds; silence then sounds like the agent did
not hear. :class:`~my_stt_tts.eleven_voice.VoiceSession` starts a timer with every client
tool call and, when the tool has not answered after ``MAC_VOICE_WAIT_AFTER_S`` (default 2),
plays :data:`WAIT_PHRASE` (``MAC_VOICE_WAIT_PHRASE``; empty switches it off). The clip is
rendered once with macOS ``say`` into the state dir and played through the call's own
output, so the echo canceller removes it from the mic like the agent's voice. The voice is
the Mac's, not the agent's: the agent cannot be made to speak while it waits for a tool.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import wave
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger("my_stt_tts.eleven_voice")

SAMPLE_RATE = 16000  # the call's PCM format: 16-bit mono 16 kHz
#: A client tool that has not answered after this many seconds gets a spoken "One moment."
WAIT_AFTER_S = 2.0
WAIT_AFTER_ENV = "MAC_VOICE_WAIT_AFTER_S"
WAIT_PHRASE = "One moment."
WAIT_PHRASE_ENV = "MAC_VOICE_WAIT_PHRASE"
WAIT_REPEAT_S = 6.0  # overlapping slow tools say it once
STATE_DIR = Path(os.environ.get("MAC_VOICE_STATE_DIR", Path.home() / ".local/state/mac-voice"))


def wait_settings(env: Mapping[str, str] | None = None) -> tuple[float, str]:
    """(seconds, phrase) of the "one moment" hint; an empty phrase switches it off."""
    env = os.environ if env is None else env
    try:
        after = float(env.get(WAIT_AFTER_ENV) or WAIT_AFTER_S)
    except ValueError:
        after = WAIT_AFTER_S
    phrase = env.get(WAIT_PHRASE_ENV, WAIT_PHRASE).strip()
    return max(0.0, after), phrase


def render_phrase(
    text: str,
    cache_dir: Path = STATE_DIR,
    runner: Callable[..., Any] = subprocess.run,
) -> bytes | None:
    """``text`` as 16 kHz mono 16-bit PCM spoken by macOS ``say`` (cached); None on failure.

    The clip is played through the call's own output, so the echo canceller removes it
    from the mic like the agent's voice.
    """
    if not text:
        return None
    digest = hashlib.sha256(text.encode()).hexdigest()[:12]
    path = cache_dir / f"phrase-{digest}.wav"
    try:
        if not path.is_file():
            cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.wav")
            runner(
                [
                    "/usr/bin/say",
                    "-o",
                    str(tmp),
                    "--file-format=WAVE",
                    f"--data-format=LEI16@{SAMPLE_RATE}",
                    text,
                ],
                check=True,
                capture_output=True,
                timeout=15,
            )
            tmp.replace(path)
        with wave.open(str(path)) as clip:
            if (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) != (
                1,
                2,
                SAMPLE_RATE,
            ):
                return None
            return clip.readframes(clip.getnframes())
    except (OSError, EOFError, wave.Error, subprocess.SubprocessError):
        log.debug("could not render %r", text, exc_info=True)
        return None
