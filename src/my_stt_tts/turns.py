"""Turn source: cut the call's mic stream into utterances and bind transcripts to them.

The ElevenLabs SDK hands us a user transcript as plain text, with no audio and no
utterance id. To check WHO said it, :class:`TurnSource` segments the same 16 kHz float32
mic frames the agent receives (``aec.VoiceProcessingDuplex.mic_frames()``) with a voice
activity detector plus :class:`~my_stt_tts.vad.SilenceEndpointer` into
:class:`Utterance` objects kept in a 20 s ring buffer. :meth:`TurnSource.bind` then maps
a transcript to the latest FINISHED utterance that ended at most :data:`BIND_WINDOW_S`
before the transcript arrived. More than one candidate, or an utterance still running
when the transcript arrived, gives :class:`Ambiguous` — every candidate must then pass
the identity check (fail closed).

Times are monotonic seconds (``time.monotonic`` by default, injectable for tests).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .vad import SilenceEndpointer

SAMPLE_RATE = 16000
# Transcripts arrived 1.0–2.4 s after the endpointer closed the utterance (spike
# S-AUDIO); 5 s keeps one candidate per turn. A constant to re-tune after the full run.
BIND_WINDOW_S = 5.0
MIN_UTTERANCE_S = 0.4  # shorter utterances are too short to judge the speaker
RING_S = 20.0
SILENCE_S = 0.5  # trailing silence that closes an utterance


class SpeechDetector(Protocol):
    """Per-frame speech flag (:class:`my_stt_tts.vad.SileroVad` or a test fake)."""

    def is_speech(self, frame: Any) -> bool: ...


@dataclass(frozen=True)
class Utterance:
    """One stretch of speech: ``start``/``end`` monotonic seconds, ``pcm`` 16 kHz float32."""

    seq: int
    start: float
    end: float
    pcm: np.ndarray = field(repr=False, compare=False)
    finished: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Ambiguous:
    """Several utterances may have produced the transcript; all of them must pass."""

    candidates: tuple[Utterance, ...]


Binding = Utterance | Ambiguous | None


@dataclass
class _Open:
    """The utterance being recorded right now."""

    start: float
    last_speech_end: float
    chunks: list[np.ndarray] = field(default_factory=list)
    pending: list[np.ndarray] = field(default_factory=list)  # silence after the last speech


class TurnSource:  # pylint: disable=too-many-instance-attributes  # segmenter + ring state
    """Segment mic frames into utterances (ring buffer) and bind transcripts to them."""

    def __init__(
        self,
        vad: SpeechDetector,
        *,
        sample_rate: int = SAMPLE_RATE,
        silence_s: float = SILENCE_S,
        clock: Callable[[], float] = time.monotonic,
        ring_s: float = RING_S,
    ) -> None:
        self.vad = vad
        self.sample_rate = sample_rate
        self.clock = clock
        self.ring_s = ring_s
        self._endpointer = SilenceEndpointer(silence_seconds=silence_s, frame_seconds=0.0)
        self._ring: deque[Utterance] = deque()
        self._open: _Open | None = None
        self._seq = 0
        self._lock = threading.Lock()

    # -- segmentation ---------------------------------------------------------------------
    def feed(self, frame: Any, at: float | None = None) -> Utterance | None:
        """Add one mic frame that ENDED at ``at`` (default: now); the utterance it closed."""
        pcm = np.asarray(frame, dtype=np.float32).ravel()
        if pcm.size == 0:
            return None
        end = self.clock() if at is None else at
        start = end - pcm.size / self.sample_rate
        speech = bool(self.vad.is_speech(pcm))
        with self._lock:
            closed = self._step(pcm, start, end, speech)
            self._prune(end)
        return closed

    def _step(self, pcm: np.ndarray, start: float, end: float, speech: bool) -> Utterance | None:
        self._endpointer.frame_seconds = end - start
        done = self._endpointer.update(speech)
        if speech:
            if self._open is None:
                self._open = _Open(start=start, last_speech_end=end)
            self._open.chunks.extend(self._open.pending)
            self._open.pending.clear()
            self._open.chunks.append(pcm)
            self._open.last_speech_end = end
        elif self._open is not None:
            self._open.pending.append(pcm)
        if done and self._open is not None:
            return self._close()
        return None

    def _close(self) -> Utterance:
        assert self._open is not None
        utt = self._make(self._open, finished=True)
        self._ring.append(utt)
        self._open = None
        self._endpointer.reset()
        return utt

    def _make(self, rec: _Open, *, finished: bool) -> Utterance:
        self._seq += 1
        pcm = np.concatenate(rec.chunks) if rec.chunks else np.zeros(0, dtype=np.float32)
        return Utterance(self._seq, rec.start, rec.last_speech_end, pcm, finished)

    def _prune(self, now: float) -> None:
        while self._ring and self._ring[0].end < now - self.ring_s:
            self._ring.popleft()

    def utterances(self) -> list[Utterance]:
        """The finished utterances still in the ring buffer, oldest first."""
        with self._lock:
            return list(self._ring)

    # -- binding --------------------------------------------------------------------------
    def bind(
        self, transcript_seq: int, text: str, received_at: float, window_s: float = BIND_WINDOW_S
    ) -> Binding:
        """The utterance behind a transcript received at ``received_at`` (see module doc).

        ``transcript_seq`` and ``text`` identify the transcript for the caller's records;
        the binding itself only depends on timing.
        """
        del transcript_seq, text  # timing alone binds; kept for the interface contract
        with self._lock:
            finished = [u for u in self._ring if received_at - window_s <= u.end <= received_at]
            running = self._running_at(received_at)
        if not finished:
            # The ASR can finalise before the local endpointer closes the utterance
            # (S-AUDIO: once 1.2 s early) — then the still-running one is the speech.
            return running
        latest = finished[-1]
        if len(finished) == 1 and running is None:
            return latest
        extra = (running,) if running is not None else ()
        return Ambiguous(tuple(finished) + extra)

    def _running_at(self, received_at: float) -> Utterance | None:
        """An utterance that had started but not ended when the transcript arrived."""
        late = [u for u in self._ring if u.start <= received_at < u.end]
        if late:
            return late[0]
        rec = self._open
        if rec is None or rec.start > received_at:
            return None
        pcm = np.concatenate(rec.chunks) if rec.chunks else np.zeros(0, dtype=np.float32)
        return Utterance(0, rec.start, rec.last_speech_end, pcm, finished=False)
