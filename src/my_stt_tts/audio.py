"""Audio capture, pre-roll ring buffer, playback, half-duplex mic gating, and
wake-word / VAD-gated capture helpers.

``sounddevice`` is lazy-imported (needs PortAudio, from the ``audio`` extra); the
buffer and gate logic are pure and unit-tested. The streaming helpers
(``record_*`` and ``listen_for_wake``) need a real mic, so they are not unit-tested.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.audio")


class PreRollBuffer:
    """Fixed-duration ring buffer of the most recent audio (float32 mono).

    Prepended on record-start so the first syllable is never clipped.
    """

    def __init__(self, sample_rate: int, seconds: float) -> None:
        self.maxlen = max(1, int(sample_rate * seconds))
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, frame: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, frame.astype(np.float32)])[-self.maxlen :]

    def get(self) -> np.ndarray:
        return self._buf.copy()

    def clear(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)


class MicGate:
    """Half-duplex gate: the mic is 'open' unless gated during playback.

    Re-opens only after a short tail so the speaker's decay/reverb is not
    recorded as user speech.
    """

    def __init__(self, tail_seconds: float = 0.2) -> None:
        self._tail = tail_seconds
        self._gated = threading.Event()
        self._timer: threading.Timer | None = None

    @property
    def open(self) -> bool:
        return not self._gated.is_set()

    def gate(self) -> None:
        """Close the mic immediately (call before starting playback)."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._gated.set()

    def release(self) -> None:
        """Re-open the mic after the configured tail (call after playback)."""
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(self._tail, self._gated.clear)
        self._timer.daemon = True
        self._timer.start()


def _sd() -> Any:  # noqa: ANN401 — thin lazy accessor
    import sounddevice as sd

    return sd


def play(samples: np.ndarray, sample_rate: int) -> None:
    """Play a float32 mono array and block until done."""
    sd = _sd()
    sd.play(samples, samplerate=sample_rate)
    sd.wait()


def record_push_to_talk(
    sample_rate: int,
    max_seconds: float,
    preroll: PreRollBuffer | None = None,
    prompt: str = "[Enter] start / stop recording: ",
) -> np.ndarray:
    """Record between two Enter presses (deterministic v1 endpointing)."""
    sd = _sd()
    input(prompt)
    frames: list[np.ndarray] = []
    stop = threading.Event()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames.append(indata[:, 0].copy())

    def _wait_for_stop() -> None:
        input()
        stop.set()

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=_callback):
        waiter = threading.Thread(target=_wait_for_stop, daemon=True)
        waiter.start()
        stop.wait(timeout=max_seconds)

    captured = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    if preroll is not None:
        captured = np.concatenate([preroll.get(), captured])
    return captured


def record_with_vad(
    sample_rate: int,
    vad: Any,
    endpointer: Any,
    *,
    max_seconds: float,
    preroll: PreRollBuffer | None = None,
    frame_samples: int = 512,
) -> np.ndarray:
    """Stream from the mic until the endpointer signals end-of-turn (or timeout).

    ``vad.is_speech(frame)`` drives ``endpointer.update(...)`` (see :mod:`.vad`).
    """
    sd = _sd()
    frames_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames_q.put(indata[:, 0].copy())

    collected: list[np.ndarray] = []
    spoke = False
    endpointer.reset()
    start = time.monotonic()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_samples,
        callback=_callback,
    ):
        while time.monotonic() - start < max_seconds:
            try:
                frame = frames_q.get(timeout=0.1)
            except queue.Empty:
                continue
            collected.append(frame)
            is_speech = vad.is_speech(frame)
            spoke = spoke or is_speech
            if endpointer.update(is_speech):
                break

    if not spoke:
        return np.zeros(0, dtype=np.float32)  # nothing was said -> end the turn
    captured = np.concatenate(collected)
    if preroll is not None:
        captured = np.concatenate([preroll.get(), captured])
    return captured


@dataclass
class BargeInResult:
    """Outcome of monitoring the mic during playback (G1).

    ``interrupted`` is True if the user barged in; ``captured`` holds the audio
    recorded from the moment speech was detected (so the new turn is not clipped).
    """

    interrupted: bool
    captured: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


def monitor_during_playback(
    playback: Any,
    sample_rate: int,
    vad: Any,
    gate: Any,
    *,
    energy_floor: float = 0.0,
    poll_seconds: float = 0.05,
    frame_samples: int = 512,
) -> BargeInResult:
    """Keep the mic LIVE while ``playback`` runs, abort it on confirmed speech.

    Runs VAD on the live input and feeds the false-interrupt :class:`gate`
    (:class:`~my_stt_tts.interrupt.InterruptGate`). A frame below ``energy_floor``
    is treated as silence to reject open-speaker bleed. When the gate authorises
    an interruption, ``playback.cancel()`` is called and the captured speech (from
    the first voiced frame onward) is returned for the user's new turn. If
    playback finishes first, returns ``interrupted=False``.

    This replaces half-duplex gating: the mic stays open the whole time.
    """
    from .interrupt import frame_energy

    sd = _sd()
    frames_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames_q.put(indata[:, 0].copy())

    gate.reset()
    captured: list[np.ndarray] = []
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_samples,
        callback=_callback,
    ):
        while not playback.done:
            try:
                frame = frames_q.get(timeout=poll_seconds)
            except queue.Empty:
                continue
            loud_enough = frame_energy(frame) >= energy_floor
            is_speech = loud_enough and vad.is_speech(frame)
            if is_speech or captured:
                captured.append(frame)  # buffer from the first voiced frame on
            if gate.update(is_speech):
                playback.cancel()
                clip = np.concatenate(captured) if captured else np.zeros(0, dtype=np.float32)
                return BargeInResult(interrupted=True, captured=clip)
    playback.wait()
    return BargeInResult(interrupted=False)


def record_turn(
    sample_rate: int,
    vad: Any,
    analyzer: Any,
    *,
    max_seconds: float,
    streamer: Any = None,
    on_partial: Any = None,
    preroll: PreRollBuffer | None = None,
    frame_samples: int = 512,
) -> np.ndarray:
    """Like :func:`record_with_vad` but driven by a :class:`TurnAnalyzer` (G2) and
    able to emit partial transcripts during the turn (G6).

    ``analyzer.update(frame, is_speech)`` decides end-of-turn (prosody-aware when
    a Smart Turn analyzer is configured). If ``streamer`` (a
    :class:`~my_stt_tts.stt.StreamingTranscriber`) is given, each frame is fed to
    it and any new partial is passed to ``on_partial``.
    """
    sd = _sd()
    frames_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames_q.put(indata[:, 0].copy())

    collected: list[np.ndarray] = []
    spoke = False
    analyzer.reset()
    if streamer is not None:
        streamer.reset()
    start = time.monotonic()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_samples,
        callback=_callback,
    ):
        while time.monotonic() - start < max_seconds:
            try:
                frame = frames_q.get(timeout=0.1)
            except queue.Empty:
                continue
            collected.append(frame)
            is_speech = vad.is_speech(frame)
            spoke = spoke or is_speech
            if streamer is not None:
                partial = streamer.feed(frame)
                if partial is not None and on_partial is not None:
                    on_partial(partial)
            if analyzer.update(frame, is_speech):
                break

    if not spoke:
        return np.zeros(0, dtype=np.float32)
    captured = np.concatenate(collected)
    if preroll is not None:
        captured = np.concatenate([preroll.get(), captured])
    return captured


def listen_for_wake(
    wake: Any, sample_rate: int, *, frame_samples: int = 1280, poll_seconds: float = 0.1
) -> None:
    """Block until ``wake.detect(frame)`` fires on an 80 ms (1280-sample) frame."""
    sd = _sd()
    frames_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames_q.put(indata[:, 0].copy())

    wake.reset()
    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_samples,
        callback=_callback,
    ):
        while True:
            try:
                frame = frames_q.get(timeout=poll_seconds)
            except queue.Empty:
                continue
            if wake.detect(frame):
                return
