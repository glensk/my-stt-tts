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


def stitch_partial(committed: str, window_text: str) -> str:
    """Join a committed prefix with the latest window decode, de-duplicating overlap.

    ``committed`` is the transcript of audio that has already scrolled OUT of the
    sliding window; ``window_text`` is the fresh decode of the window (which still
    overlaps the tail of the committed audio). We drop the longest suffix of
    ``committed`` that is a (word-aligned) prefix of ``window_text`` so the
    overlap isn't doubled, then concatenate. With no committed text yet, the
    window decode is the whole partial.
    """
    committed = committed.strip()
    window_text = window_text.strip()
    if not committed:
        return window_text
    if not window_text:
        return committed
    cwords = committed.split()
    wwords = window_text.split()
    # Find the largest k such that the last k committed words == the first k window
    # words; that overlap is the audio shared by both, kept once from the window.
    max_k = min(len(cwords), len(wwords))
    overlap = 0
    for k in range(max_k, 0, -1):
        if cwords[-k:] == wwords[:k]:
            overlap = k
            break
    stitched = cwords[: len(cwords) - overlap] + wwords
    return " ".join(stitched)


class StreamingTranscriber:
    """Bounded sliding-window incremental transcription (G6 / R2-2).

    parakeet-mlx has no token-streaming API, so streaming is approximated by
    re-running the engine. The naive G6 implementation re-decoded the ENTIRE
    growing buffer every interval, so partial latency and per-call CPU grew with
    the utterance length. This version re-decodes only the **last ``window_s``
    seconds** of audio and *stitches* that window decode onto a committed prefix
    (the transcript of audio that has already scrolled out of the window), so each
    partial costs at most one bounded-length decode regardless of how long the
    user talks. When the buffer is still shorter than the window, behaviour is
    identical to re-decoding the whole clip.

    Engine-agnostic: any object with ``transcribe(audio, sample_rate)`` works, so
    tests can inject a fake without a mic or GPU.
    """

    def __init__(
        self,
        engine: Transcriber,
        sample_rate: int = 16000,
        *,
        partial_interval_ms: float = 600.0,
        window_s: float = 7.0,
    ) -> None:
        self.engine = engine
        self.sample_rate = sample_rate
        self.partial_interval_ms = partial_interval_ms
        self.window_s = window_s
        self._chunks: list[np.ndarray] = []
        self._total_samples = 0
        self._samples_since_partial = 0
        self._last_partial = ""
        # Transcript of audio that has scrolled out of the re-decode window, plus
        # the sample offset up to which it is committed.
        self._committed = ""
        self._committed_samples = 0

    @property
    def _interval_samples(self) -> int:
        return max(1, int(self.sample_rate * self.partial_interval_ms / 1000.0))

    @property
    def _window_samples(self) -> int:
        return max(1, int(self.sample_rate * self.window_s))

    def reset(self) -> None:
        """Clear buffered audio for a new turn."""
        self._chunks = []
        self._total_samples = 0
        self._samples_since_partial = 0
        self._last_partial = ""
        self._committed = ""
        self._committed_samples = 0

    def feed(self, frame: np.ndarray) -> str | None:
        """Add a mic frame; return a NEW partial transcript when one is due, else None.

        A partial is produced once at least ``partial_interval_ms`` of audio has
        accumulated since the previous one, and only if the text actually changed.
        Only the last ``window_s`` of audio is re-decoded per partial (R2-2).
        """
        arr = np.asarray(frame, dtype=np.float32).ravel()
        if arr.size:
            self._chunks.append(arr)
            self._total_samples += arr.size
            self._samples_since_partial += arr.size
        if self._samples_since_partial < self._interval_samples:
            return None
        self._samples_since_partial = 0
        text = self._partial_text()
        if text and text != self._last_partial:
            self._last_partial = text
            return text
        return None

    def feed_clip(self, clip: np.ndarray) -> None:
        """Seed the buffer with already-captured audio without emitting a partial.

        Used on a barge-in (R2-6): the audio captured while the bot was speaking is
        handed straight to the streamer for the *next* turn, so it does not have to
        be re-transcribed from scratch — subsequent live frames just extend it.
        """
        arr = np.asarray(clip, dtype=np.float32).ravel()
        if arr.size:
            self._chunks.append(arr)
            self._total_samples += arr.size
            self._samples_since_partial += arr.size

    def _audio(self) -> np.ndarray:
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        if len(self._chunks) > 1:  # coalesce so we don't reconcatenate every call
            self._chunks = [np.concatenate(self._chunks)]
        return self._chunks[0]

    def _partial_text(self) -> str:
        """Bounded re-decode of the trailing window, stitched onto the committed prefix.

        As audio scrolls past the window, the prefix that has fallen OUT of the
        window is decoded once and folded into ``_committed`` so each partial only
        re-decodes a bounded window. The committed boundary is advanced in
        half-window steps and overlaps the live window slightly so the stitch can
        de-duplicate cleanly.
        """
        audio = self._audio()
        if audio.size == 0:
            return ""
        window_n = self._window_samples
        if audio.size <= window_n:
            # Whole utterance still fits the window: decode it directly.
            return self.engine.transcribe(audio, self.sample_rate).text.strip()
        # Advance the committed boundary so the live window stays bounded. Commit in
        # half-window steps; keep a half-window of overlap with the live window.
        window_start = audio.size - window_n
        step = max(1, window_n // 2)
        while self._committed_samples + step <= window_start:
            seg = audio[self._committed_samples : self._committed_samples + step]
            seg_text = self.engine.transcribe(seg, self.sample_rate).text.strip()
            self._committed = stitch_partial(self._committed, seg_text)
            self._committed_samples += step
        window = audio[self._committed_samples :]
        window_text = self.engine.transcribe(window, self.sample_rate).text.strip()
        return stitch_partial(self._committed, window_text)

    def final(self) -> STTResult:
        """Transcribe the full accumulated buffer (end-of-turn).

        Decodes the whole utterance in one pass for maximum accuracy (the windowed
        partials were for latency only); detected language is preserved.
        """
        audio = self._audio()
        if audio.size == 0:
            return STTResult(text="")
        return self.engine.transcribe(audio, self.sample_rate)


def stream_transcribe(
    engine: Transcriber,
    frames: Iterator[np.ndarray],
    sample_rate: int = 16000,
    *,
    partial_interval_ms: float = 600.0,
    window_s: float = 7.0,
    on_partial: Callable[[str], None] | None = None,
) -> STTResult:
    """Drive a :class:`StreamingTranscriber` over ``frames``; return the final.

    ``on_partial`` (if given) is called with each new partial transcript as it
    becomes available — wire it to ``bus.transcript(text, partial=True)``.
    """
    streamer = StreamingTranscriber(
        engine, sample_rate, partial_interval_ms=partial_interval_ms, window_s=window_s
    )
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
