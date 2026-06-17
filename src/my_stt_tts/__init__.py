"""my-stt-tts: a local voice-assistant loop for macOS Apple Silicon.

Pipeline: wake word -> speech-to-text -> an LLM (streaming) -> text-to-speech ->
playback, with speaker identification and German/French/English support.

The orchestrator (this package) is pure Python and imports without the heavy ML
backends; those are lazy-imported inside the backend modules and installed via
the optional extras (see ``pyproject.toml``).
"""

__version__ = "0.0.1"
