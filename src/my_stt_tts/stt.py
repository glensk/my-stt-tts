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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

log = logging.getLogger("my_stt_tts.stt")


@dataclass
class STTResult:
    """A transcript plus an optional ISO-639-1 language code."""

    text: str
    language: str | None = None


class Transcriber(Protocol):
    """Minimal STT engine surface used by the streaming path."""

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        """Transcribe a float32 mono clip."""
        ...


class StreamingTranscriber:
    """Incremental transcription: re-transcribe the growing audio buffer (G6).

    parakeet-mlx has no token-streaming API, so we approximate streaming by
    re-running the engine on the **accumulated** audio every
    ``partial_interval_ms`` of new audio and emitting the result as a *partial*
    transcript. The caller publishes partials to the event bus during the turn
    and gets a final on end-of-turn — cutting perceived latency versus waiting
    for the whole clip. Re-transcription is cheap on MLX for short turns and is
    skipped when an interval hasn't elapsed.

    Engine-agnostic: any object with ``transcribe(audio, sample_rate)`` works,
    so tests can inject a fake without a mic or GPU.
    """

    def __init__(
        self,
        engine: Transcriber,
        sample_rate: int = 16000,
        *,
        partial_interval_ms: float = 600.0,
    ) -> None:
        self.engine = engine
        self.sample_rate = sample_rate
        self.partial_interval_ms = partial_interval_ms
        self._chunks: list[np.ndarray] = []
        self._samples_since_partial = 0
        self._last_partial = ""

    @property
    def _interval_samples(self) -> int:
        return max(1, int(self.sample_rate * self.partial_interval_ms / 1000.0))

    def reset(self) -> None:
        """Clear buffered audio for a new turn."""
        self._chunks = []
        self._samples_since_partial = 0
        self._last_partial = ""

    def feed(self, frame: np.ndarray) -> str | None:
        """Add a mic frame; return a NEW partial transcript when one is due, else None.

        A partial is produced once at least ``partial_interval_ms`` of audio has
        accumulated since the previous one, and only if the text actually changed.
        """
        arr = np.asarray(frame, dtype=np.float32).ravel()
        if arr.size:
            self._chunks.append(arr)
            self._samples_since_partial += arr.size
        if self._samples_since_partial < self._interval_samples:
            return None
        self._samples_since_partial = 0
        text = self._transcribe_all().text.strip()
        if text and text != self._last_partial:
            self._last_partial = text
            return text
        return None

    def final(self) -> STTResult:
        """Transcribe the full accumulated buffer (end-of-turn)."""
        return self._transcribe_all()

    def _transcribe_all(self) -> STTResult:
        if not self._chunks:
            return STTResult(text="")
        audio = np.concatenate(self._chunks)
        return self.engine.transcribe(audio, self.sample_rate)


def stream_transcribe(
    engine: Transcriber,
    frames: Iterator[np.ndarray],
    sample_rate: int = 16000,
    *,
    partial_interval_ms: float = 600.0,
    on_partial: Callable[[str], None] | None = None,
) -> STTResult:
    """Drive a :class:`StreamingTranscriber` over ``frames``; return the final.

    ``on_partial`` (if given) is called with each new partial transcript as it
    becomes available — wire it to ``bus.transcript(text, partial=True)``.
    """
    streamer = StreamingTranscriber(engine, sample_rate, partial_interval_ms=partial_interval_ms)
    for frame in frames:
        partial = streamer.feed(frame)
        if partial is not None and on_partial is not None:
            on_partial(partial)
    return streamer.final()


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
