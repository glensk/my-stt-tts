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
import queue
import tempfile
import threading
import wave
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import numpy as np

log = logging.getLogger("my_stt_tts.stt")

_T = TypeVar("_T")


class _STTWorker:
    """A single long-lived thread that owns the MLX engine and runs ALL transcription.

    parakeet-mlx loads its model — and creates its GPU (Metal) stream — on whatever
    thread first touches it. MLX streams have **thread affinity**: a call from any
    OTHER thread raises ``There is no Stream(gpu, 0) in current thread``. The GUI
    spawns a FRESH daemon thread per push-to-talk / mic-test / wake action (and the
    wake loop + barge-in paths run on their own threads), so the model-load thread and
    the call thread differed and PTT crashed.

    The fix: marshal every transcribe call onto ONE dedicated worker thread. Callers
    submit a closure and block for its result, so the model is always loaded on — and
    only ever called from — this single worker thread, regardless of which thread the
    caller runs on. The worker is a process-wide lazy singleton (see :func:`stt_worker`);
    work is serialized through a queue, so concurrent callers are simply queued.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[
            tuple[Callable[[], Any], queue.Queue[tuple[Any, BaseException | None]]]
        ] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="mstt-stt-worker", daemon=True)
        self._started = False
        self._start_lock = threading.Lock()
        # The id of the worker thread, exposed so a test can assert every MLX call
        # landed on the SAME thread (the affinity invariant). Set when the thread runs.
        self.thread_ident: int | None = None

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._start_lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def _run(self) -> None:
        self.thread_ident = threading.get_ident()
        while True:
            fn, reply = self._queue.get()
            try:
                result: Any = fn()
                err: BaseException | None = None
            except BaseException as exc:  # noqa: BLE001 — re-raised on the caller's thread
                result, err = None, exc
            reply.put((result, err))

    def submit(self, fn: Callable[[], _T]) -> _T:
        """Run ``fn`` on the worker thread and block for its result (re-raising errors).

        If called FROM the worker thread itself (e.g. one engine method calling
        another), run inline to avoid a self-deadlock — it is already on the right
        thread, which is exactly the affinity guarantee we want.
        """
        self._ensure_started()
        if threading.get_ident() == self.thread_ident:
            return fn()
        reply: queue.Queue[tuple[Any, BaseException | None]] = queue.Queue(maxsize=1)
        self._queue.put((fn, reply))
        result, err = reply.get()
        if err is not None:
            raise err
        return result  # type: ignore[no-any-return]


_WORKER: _STTWorker | None = None
_WORKER_LOCK = threading.Lock()


def stt_worker() -> _STTWorker:
    """The process-wide single STT worker thread (lazy singleton).

    Every MLX/parakeet transcribe call is marshalled onto this one thread so the model
    is loaded on — and only ever called from — the thread that owns its GPU stream.
    """
    global _WORKER  # noqa: PLW0603 — intentional lazy singleton
    if _WORKER is None:
        with _WORKER_LOCK:
            if _WORKER is None:
                _WORKER = _STTWorker()
    return _WORKER


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
        """Transcribe a float32 mono clip to text + detected language.

        MLX has thread-affine GPU streams: the model must be loaded on AND every
        decode must run on ONE consistent thread, or a call from a different thread
        raises ``There is no Stream(gpu, 0) in current thread`` (the PTT crash). So the
        actual model load + decode is marshalled onto the single :func:`stt_worker`
        thread; the caller (PTT/mic-test/wake/barge-in worker threads) simply blocks
        for the result. The public surface is unchanged."""
        return stt_worker().submit(lambda: self._transcribe_on_worker(audio, sample_rate))

    def _transcribe_on_worker(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        """The real MLX load + decode — ALWAYS runs on the single STT worker thread."""
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


class CloudTranscriber:
    """Optional cloud STT (R2-7): an OpenAI-compatible transcription endpoint.

    Sends the clip as a WAV to a ``/audio/transcriptions``-style API (OpenAI,
    Deepgram-compatible gateways, a local server, …). **Local-first**: this is only
    selected when ``stt_backend=cloud`` is configured *and* an API key is present;
    the orchestrator falls back to the local engine otherwise (never hard-fails on
    a missing key). The ``openai`` client is lazy-imported from the ``llm`` extra.
    """

    def __init__(
        self,
        model: str = "whisper-1",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self._client: Any = None

    def available(self) -> bool:
        """True when an API key is configured (so cloud STT can actually be used)."""
        return bool(self.api_key)

    def _ensure(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key or "not-needed", base_url=self.base_url)
        return self._client

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        """Transcribe a clip via the cloud endpoint; returns text (+ language if given)."""
        client = self._ensure()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            wav_path = handle.name
        try:
            _write_wav(wav_path, audio, sample_rate)
            with open(wav_path, "rb") as fh:  # noqa: PTH123 — SDK wants a file object
                result = client.audio.transcriptions.create(model=self.model, file=fh)
        finally:
            Path(wav_path).unlink(missing_ok=True)
        text = str(getattr(result, "text", result) or "").strip()
        language = getattr(result, "language", None)
        return STTResult(text=text, language=language)


def make_transcriber(cfg: Any) -> Transcriber:
    """Select the STT engine from config via the backend registry (G1, local-first).

    ``cfg.stt_backend`` names a registered backend (``local`` / ``whispercpp`` /
    ``faster-whisper`` / ``cloud`` / ``openai`` / ``deepgram`` …). Cloud backends
    are key-gated: a selected-but-unusable cloud backend gracefully falls back to
    the local on-device :class:`ParakeetSTT`, so a missing key never hard-fails.
    """
    from .registry import select_transcriber

    return select_transcriber(cfg)


class WhisperCppSTT:
    """Cross-platform backend: whisper.cpp via ``pywhispercpp`` (G8).

    ``pywhispercpp`` bundles whisper.cpp's C++ runtime as a wheel (Metal/CoreML on
    Apple Silicon, CPU/CUDA elsewhere), so the central "brain" can run STT on a
    Linux box without MLX — the cross-platform fallback for the parakeet-mlx
    primary, which is Apple-Silicon-only. The model is lazy-loaded from the
    ``whispercpp`` extra; multilingual whisper returns a detected language per
    segment, surfaced as :attr:`STTResult.language`.
    """

    def __init__(self, model: str = "large-v3-turbo", *, language: str | None = None) -> None:
        self.model_id = model
        self.language = language  # None => auto-detect (multilingual)
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from pywhispercpp.model import Model

            self._model = Model(self.model_id)

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        """Transcribe a float32 mono clip; reads back a detected language if exposed."""
        self._ensure()
        kwargs: dict[str, Any] = {}
        if self.language:
            kwargs["language"] = self.language
        segments = self._model.transcribe(np.asarray(audio, dtype=np.float32), **kwargs)
        seg_list = list(segments)
        text = " ".join(getattr(seg, "text", "") for seg in seg_list).strip()
        # whisper.cpp exposes the detected language on the model after a decode.
        language = self.language or getattr(self._model, "detected_language", None)
        return STTResult(text=text, language=language)


class FasterWhisperSTT:
    """Cross-platform backend: faster-whisper (CTranslate2) for Linux/CPU/CUDA (G8).

    faster-whisper runs the Whisper models on the CTranslate2 engine — fast on a
    Linux CPU or NVIDIA GPU, so a non-Mac brain host has a strong multilingual STT
    without MLX. (On a Mac it is CPU-only and slower than parakeet-mlx, so it is an
    *off-Mac* option, not the macOS default.) The model is lazy-loaded from the
    ``faster-whisper`` wheel; it returns segments plus a detected language.
    """

    def __init__(
        self,
        model: str = "large-v3-turbo",
        *,
        device: str = "auto",
        compute_type: str = "int8",
        language: str | None = None,
    ) -> None:
        self.model_id = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self._model: Any = None

    def _ensure(self) -> None:
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.model_id, device=self.device, compute_type=self.compute_type
            )

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        """Transcribe a float32 mono clip; returns text + the detected language."""
        self._ensure()
        segments, info = self._model.transcribe(
            np.asarray(audio, dtype=np.float32), language=self.language
        )
        text = " ".join(getattr(seg, "text", "") for seg in segments).strip()
        language = self.language or getattr(info, "language", None)
        return STTResult(text=text, language=language)
