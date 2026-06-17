"""Text-to-speech router: per-language voice selection + pluggable backends.

The default engine is **Piper**, invoked as a SUBPROCESS (its CLI binary) so this
Apache-2.0 project never links the GPL-3.0 library in-process. macOS ``say`` is
the always-available fallback. Language detection (``lingua``) is optional/lazy
and the detector is cached so per-sentence detection is cheap. English voices can
be swapped via :data:`VOICE_PRESETS`; missing Piper voices are downloaded on first
use, and ``tts_length_scale`` slows delivery for a calmer cadence.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

from .config import Config

log = logging.getLogger("my_stt_tts.tts")

# A small menu of English Piper voices to pick from (`--voice <name>`).
VOICE_PRESETS: dict[str, str] = {
    "lessac": "en_US-lessac-medium",
    "amy": "en_US-amy-medium",
    "ryan": "en_US-ryan-medium",
    "kristin": "en_US-kristin-medium",
    "hfc-female": "en_US-hfc_female-medium",
    "hfc-male": "en_US-hfc_male-medium",
    "joe": "en_US-joe-medium",
    "alba": "en_GB-alba-medium",
    "alan": "en_GB-alan-medium",
}

_VOICE_NOTES: dict[str, str] = {
    "lessac": "neutral, clear (default)",
    "amy": "calm female",
    "ryan": "calm male",
    "kristin": "warm female",
    "hfc-female": "natural female",
    "hfc-male": "natural male",
    "joe": "deep male",
    "alba": "British female",
    "alan": "British male",
}


def list_voice_presets() -> str:
    """Return a printable menu of the English voice presets."""
    return "\n".join(
        f"  {name:11} {VOICE_PRESETS[name]:24} {_VOICE_NOTES.get(name, '')}"
        for name in VOICE_PRESETS
    )


@lru_cache(maxsize=4)
def _detector(langs: tuple[str, ...]):  # noqa: ANN202 — lingua type is lazy-imported
    from lingua import IsoCode639_1, LanguageDetectorBuilder

    codes = [getattr(IsoCode639_1, c.upper()) for c in langs]
    return LanguageDetectorBuilder.from_iso_codes_639_1(*codes).build()


def detect_language(
    text: str, default: str = "en", langs: tuple[str, ...] = ("de", "fr", "en")
) -> str:
    """Detect the language of ``text`` (lingua); fall back to ``default``."""
    try:
        detected = _detector(langs).detect_language_of(text)
    except ImportError:
        return default  # lingua not installed (the `lang` extra)
    except Exception:  # any build/detection error must never break the loop
        log.debug("language detection unavailable; using default %r", default, exc_info=True)
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


def _ensure_piper_voice(data_dir: str, voice: str) -> bool:
    """Make sure ``<data_dir>/<voice>.onnx`` exists, downloading it if needed."""
    if (Path(data_dir) / f"{voice}.onnx").exists():
        return True
    log.info("downloading Piper voice %s ...", voice)
    subprocess.run(  # noqa: S603, S607
        [
            "uv",
            "tool",
            "run",
            "--from",
            "piper-tts",
            "python",
            "-m",
            "piper.download_voices",
            "--download-dir",
            data_dir,
            voice,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return (Path(data_dir) / f"{voice}.onnx").exists()


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
        if (
            engine == "piper"
            and shutil.which("piper")
            and _ensure_piper_voice(self.cfg.piper_data_dir, voice)
        ):
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
            cmd = [
                "piper",
                "-m",
                voice,
                "-f",
                out,
                "--length-scale",
                str(self.cfg.tts_length_scale),
            ]
            if self.cfg.piper_data_dir:
                cmd += ["--data-dir", self.cfg.piper_data_dir]
            subprocess.run(cmd, input=text.encode(), check=True)  # noqa: S603, S607
            _afplay(out)
        finally:
            Path(out).unlink(missing_ok=True)

    def _speak_say(self, text: str, voice: str) -> None:
        cmd = ["say"]
        if voice:
            cmd += ["-v", voice]
        cmd.append(text)
        subprocess.run(cmd, check=False)  # noqa: S603, S607
