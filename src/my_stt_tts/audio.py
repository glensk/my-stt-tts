"""Audio capture, pre-roll ring buffer, playback, half-duplex mic gating, and
wake-word / VAD-gated capture helpers.

``sounddevice`` is lazy-imported (needs PortAudio, from the ``audio`` extra); the
buffer and gate logic are pure and unit-tested. The streaming helpers
(``record_*`` and ``listen_for_wake``) need a real mic, so they are not unit-tested.
"""

from __future__ import annotations

import contextlib
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


def play(samples: np.ndarray, sample_rate: int, cfg: Any = None) -> None:
    """Play a float32 mono array and block until done (cross-platform, G8).

    Delegates to :func:`my_stt_tts.platform.play_array`, which prefers
    ``sounddevice`` (macOS/Linux/Windows) and falls back to a Linux CLI player
    (``aplay`` / ``paplay``) on a host without PortAudio. ``cfg`` (optional) lets a
    Linux brain pin a specific playback backend; without it the macOS behaviour is
    unchanged.
    """
    from .platform import play_array

    play_array(samples, sample_rate, cfg)


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
    aec: Any = None,
    predictor: Any = None,
    source: Any = None,
    denoiser: Any = None,
) -> BargeInResult:
    """Keep the mic LIVE while ``playback`` runs, abort it on confirmed speech.

    Runs VAD on the live input and feeds the false-interrupt :class:`gate`
    (:class:`~my_stt_tts.interrupt.InterruptGate`). A frame below ``energy_floor``
    is treated as silence to reject open-speaker bleed. When the gate authorises
    an interruption, ``playback.cancel()`` is called and the captured speech (from
    the first voiced frame onward) is returned for the user's new turn. If
    playback finishes first, returns ``interrupted=False``.

    ``aec`` (an :class:`~my_stt_tts.aec.EchoCanceller`, R2-1) is fed the playback
    reference signal and run on each mic frame so the assistant's own voice is
    removed before VAD — this is what makes barge-in work on open speakers. When
    the canceller is active the ``energy_floor`` is relaxed (echo is already gone),
    so genuine speech below the open-speaker-bleed floor is no longer suppressed.

    ``predictor`` (an :class:`~my_stt_tts.interrupt.InterruptPredictor`, R2-3) scores
    each cancelled frame for *intent to take the floor* and composes its verdict
    with the gate, so a sustained real interruption can win even before two words
    transcribe while backchannels still talk through.

    This replaces half-duplex gating: the mic stays open the whole time.
    """
    from .interrupt import frame_energy

    gate.reset()
    if aec is not None:
        aec.reset()
        _seed_aec_reference(aec, playback, sample_rate)
    if predictor is not None:
        predictor.reset()
    if denoiser is not None:
        denoiser.reset()
    # With echo cancelled by an active AEC, the bleed floor is no longer needed;
    # relax it so real (possibly quiet) speech is not gated out as bleed. The HW
    # capture source (R3-4) already delivers OS-cancelled audio, so treat it as an
    # active AEC for the floor too.
    hw_cancelled = source is not None and aec is None
    aec_active = (aec is not None and getattr(aec, "active", False)) or hw_cancelled
    effective_floor = 0.0 if aec_active else energy_floor
    captured: list[np.ndarray] = []

    def _on_frame(frame: np.ndarray) -> BargeInResult | None:
        clean = aec.process(frame) if aec is not None else frame
        if denoiser is not None:
            clean = denoiser.process(clean)
        loud_enough = frame_energy(clean) >= effective_floor
        is_speech = bool(loud_enough and vad.is_speech(clean))
        if is_speech or captured:
            captured.append(clean)  # buffer the echo-cancelled, voiced audio
        predicted = predictor.update(clean, is_speech) if predictor is not None else False
        if gate.update(is_speech) or predicted:
            playback.cancel()
            clip = np.concatenate(captured) if captured else np.zeros(0, dtype=np.float32)
            return BargeInResult(interrupted=True, captured=clip)
        return None

    if source is not None:
        # R3-4: pull HW-cancelled frames from the VoiceProcessingIO capture source.
        for frame in source.mic_frames():
            if playback.done:
                break
            res = _on_frame(np.asarray(frame, dtype=np.float32).ravel())
            if res is not None:
                return res
        playback.wait()
        return BargeInResult(interrupted=False)

    sd = _sd()
    frames_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames_q.put(indata[:, 0].copy())

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
            res = _on_frame(frame)
            if res is not None:
                return res
    playback.wait()
    return BargeInResult(interrupted=False)


def _seed_aec_reference(aec: Any, playback: Any, sample_rate: int) -> None:
    """Prime the echo canceller with the playback's reference PCM, if any (R2-1).

    Software AEC needs the played-back signal to subtract; ``Playback`` exposes it
    as ``.reference`` (resampled to ``sample_rate`` by nearest-sample decimation if
    the player ran at a different rate). Best-effort: never raises.
    """
    ref = getattr(playback, "reference", None)
    if ref is None:
        return
    ref = np.asarray(ref, dtype=np.float32).ravel()
    ref_sr = getattr(playback, "reference_sr", None) or sample_rate
    if ref_sr != sample_rate and ref.size:
        idx = (np.arange(int(ref.size * sample_rate / ref_sr)) * ref_sr / sample_rate).astype(int)
        idx = idx[idx < ref.size]
        ref = ref[idx]
    with contextlib.suppress(Exception):
        aec.push_reference(ref)


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
    source: Any = None,
    denoiser: Any = None,
) -> np.ndarray:
    """Like :func:`record_with_vad` but driven by a :class:`TurnAnalyzer` (G2) and
    able to emit partial transcripts during the turn (G6).

    ``analyzer.update(frame, is_speech)`` decides end-of-turn (prosody-aware when
    a Smart Turn analyzer is configured). If ``streamer`` (a
    :class:`~my_stt_tts.stt.StreamingTranscriber`) is given, each frame is fed to
    it and any new partial is passed to ``on_partial``. ``source`` (R3-4) sources
    HW-AEC-cancelled frames from a :class:`~my_stt_tts.aec.VoiceProcessingCapture`
    instead of opening a ``sounddevice`` stream; ``denoiser`` (R3-6) cleans each
    frame before VAD/STT.
    """
    collected: list[np.ndarray] = []
    spoke = False
    analyzer.reset()
    if streamer is not None:
        streamer.reset()
    if denoiser is not None:
        denoiser.reset()

    def _on_frame(raw: np.ndarray) -> bool:
        nonlocal spoke
        frame = denoiser.process(raw) if denoiser is not None else raw
        collected.append(frame)
        is_speech = vad.is_speech(frame)
        spoke = spoke or is_speech
        if streamer is not None:
            partial = streamer.feed(frame)
            if partial is not None and on_partial is not None:
                on_partial(partial)
        return bool(analyzer.update(frame, is_speech))

    start = time.monotonic()
    if source is not None:
        for raw in source.mic_frames():
            if time.monotonic() - start >= max_seconds:
                break
            if _on_frame(np.asarray(raw, dtype=np.float32).ravel()):
                break
    else:
        sd = _sd()
        frames_q: queue.Queue[np.ndarray] = queue.Queue()

        def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
            frames_q.put(indata[:, 0].copy())

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
                if _on_frame(frame):
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
