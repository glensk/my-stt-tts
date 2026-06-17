"""Text-to-speech router: per-language voice selection + pluggable backends.

The default engine is **Piper**, invoked as a SUBPROCESS (its CLI binary) so this
Apache-2.0 project never links the GPL-3.0 library in-process. macOS ``say`` is
the always-available fallback. Language detection (``lingua``) is optional/lazy.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from .config import Config

log = logging.getLogger("my_stt_tts.tts")


def detect_language(
    text: str, default: str = "en", langs: tuple[str, ...] = ("de", "fr", "en")
) -> str:
    """Detect the language of ``text`` (lingua); fall back to ``default``."""
    try:
        from lingua import IsoCode639_1, LanguageDetectorBuilder
    except ImportError:
        return default
    try:
        codes = [IsoCode639_1[c.upper()] for c in langs]
        detector = LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()
        detected = detector.detect_language_of(text)
    except Exception:  # detection must never break the loop
        log.exception("language detection failed; using default %r", default)
        return default
    if detected is None:
        return default
    return detected.iso_code_639_1.name.lower()


def select_voice(cfg: Config, lang: str) -> tuple[str, str]:
    """Return ``(engine, voice)``: Piper voice for ``lang`` if mapped, else ``say``."""
    if lang in cfg.tts_voices:
        return "piper", cfg.tts_voices[lang]
    if lang in cfg.say_voices:
        return "say", cfg.say_voices[lang]
    return "say", cfg.say_voices.get(cfg.default_language, "")


def _afplay(path: str) -> None:
    subprocess.run(["afplay", path], check=False)  # noqa: S603, S607


class TTSRouter:
    """Synthesize and play text in the right voice for its language."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def speak(self, text: str, lang: str | None = None) -> None:
        """Synthesize ``text`` and play it (blocking)."""
        text = text.strip()
        if not text:
            return
        lang = lang or detect_language(text, self.cfg.default_language)
        engine, voice = select_voice(self.cfg, lang)
        if engine == "piper" and shutil.which("piper"):
            self._speak_piper(text, voice)
        else:
            say_voice = self.cfg.say_voices.get(lang) or self.cfg.say_voices.get(
                self.cfg.default_language, ""
            )
            self._speak_say(text, say_voice)

    def _speak_piper(self, text: str, voice: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            out = handle.name
        try:
            subprocess.run(  # noqa: S603, S607
                ["piper", "-m", voice, "-f", out], input=text.encode(), check=True
            )
            _afplay(out)
        finally:
            Path(out).unlink(missing_ok=True)

    def _speak_say(self, text: str, voice: str) -> None:
        cmd = ["say"]
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        subprocess.run(cmd, check=False)  # noqa: S603, S607
