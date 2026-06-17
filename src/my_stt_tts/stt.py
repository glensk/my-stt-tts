"""Speech-to-text backends (parakeet-mlx primary, whisper.cpp alternate).

parakeet-mlx is MLX-native on Apple Silicon and multilingual (DE/FR/EN), so it
returns both the transcript and a detected language. Models are lazy-imported
from the ``stt`` extra.

NOTE: the exact parakeet-mlx result attribute names should be verified on-device;
this wraps the documented ``transcribe(path)`` API and reads ``.text`` /
``.language`` defensively.
"""

from __future__ import annotations

import logging
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.stt")


@dataclass
class STTResult:
    """A transcript plus an optional ISO-639-1 language code."""

    text: str
    language: str | None = None


def _write_wav(path: str, audio: np.ndarray, sample_rate: int) -> None:
    # pylint: disable=no-member  # wave.open(..., "wb") -> Wave_write; pylint infers Wave_read
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm16.tobytes())


class ParakeetSTT:
    """NVIDIA Parakeet (TDT) running on Apple MLX, multilingual."""

    def __init__(self, model: str = "mlx-community/parakeet-tdt-0.6b-v3") -> None:
        self.model_id = model
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from parakeet_mlx import from_pretrained

            self._model = from_pretrained(self.model_id)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        """Transcribe a float32 mono clip to text + detected language."""
        self._ensure()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = handle.name
        try:
            _write_wav(wav_path, audio, sample_rate)
            result = self._model.transcribe(wav_path)
        finally:
            Path(wav_path).unlink(missing_ok=True)
        text = str(getattr(result, "text", result) or "").strip()
        language = getattr(result, "language", None)
        return STTResult(text=text, language=language)


class WhisperCppSTT:
    """Alternate backend: whisper.cpp via ``pywhispercpp`` (Metal/CoreML)."""

    def __init__(self, model: str = "large-v3-turbo") -> None:
        self.model_id = model
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from pywhispercpp.model import Model

            self._model = Model(self.model_id)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        self._ensure()
        segments = self._model.transcribe(np.asarray(audio, dtype=np.float32))
        text = " ".join(seg.text for seg in segments).strip()
        return STTResult(text=text, language=None)
