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


def mic_available() -> bool:
    """True when an input (capture) device is usable for mic capture.

    Used by the browser GUI to decide whether the server-side wake / push-to-talk
    controls can actually do anything (they need a real mic via ``sounddevice``).
    Defensive: a missing ``sounddevice`` (no ``audio`` extra), no PortAudio, or no
    input device all resolve to ``False`` instead of raising. Note that on macOS the
    first real capture still triggers a Terminal microphone-permission prompt — this
    only confirms a device exists, not that permission has been granted.
    """
    try:
        sd = _sd()
        default_in = sd.default.device[0]  # (input, output) indices; -1/None if unset
        if default_in is not None and default_in >= 0:
            return True
        return any(int(dev.get("max_input_channels", 0)) > 0 for dev in sd.query_devices())
    except Exception:  # no sounddevice / no PortAudio / no device
        return False


@dataclass(slots=True)
class MicTestResult:
    """Outcome of a short server-side microphone capture (GUI "Test mic").

    ``ok`` is True only when audio was actually captured above the silence floor.
    ``level`` is the measured loudness as a 0–100 percentage (peak-based, the most
    intuitive for a UI meter); ``rms`` / ``peak`` are the raw 0–1 amplitudes.
    ``verdict`` is a short machine tag (``ok`` / ``silent`` / ``no_device`` /
    ``error``); ``message`` is the human-facing line the UI shows prominently.
    """

    ok: bool
    verdict: str
    message: str
    level: int = 0
    rms: float = 0.0
    peak: float = 0.0
    permission: str = "unknown"


def mic_permission_status() -> str:
    """The macOS microphone authorization for this app, WITHOUT capturing audio.

    Queries ``AVCaptureDevice.authorizationStatus(for: .audio)`` via PyObjC and maps
    it to ``authorized`` / ``denied`` / ``notDetermined`` / ``restricted``. The status
    is per-app (the process inherits its terminal/launcher's grant), so this answers
    "is System Settings › Privacy & Security › Microphone enabled for this app?".
    Returns ``unavailable`` off macOS or when the AVFoundation binding (the ``aec``
    extra) isn't installed — callers then fall back to the capture-based verdict.
    """
    try:
        from AVFoundation import (  # type: ignore[import-not-found]
            AVCaptureDevice,
            AVMediaTypeAudio,
        )

        status = int(AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeAudio))
    except Exception:  # noqa: BLE001 — non-macOS / no pyobjc-AVFoundation / API change
        return "unavailable"
    return {0: "notDetermined", 1: "restricted", 2: "denied", 3: "authorized"}.get(
        status, "unavailable"
    )


# Below this peak amplitude (of full-scale 1.0) a capture is treated as "no audio"
# — a silent/near-zero stream on macOS almost always means the mic permission was
# never granted to the terminal/app (the OS feeds zeros, no error).
_SILENCE_PEAK = 0.01

_MIC_PERMISSION_HINT = (
    "grant microphone permission to your terminal/app in System Settings › "
    "Privacy & Security › Microphone, then retry"
)


def mic_test_verdict(
    *,
    captured: bool,
    rms: float,
    peak: float,
    error: str | None = None,
    permission: str = "unknown",
) -> MicTestResult:
    """Map a capture outcome (+ macOS mic permission) to a user-facing result (pure).

    Split out from :func:`mic_test` so the verdict logic is unit-testable without a
    real microphone. A *conclusive* permission (``denied`` / ``restricted``) wins —
    no capture can succeed without it; otherwise ``error`` wins, then ``captured`` +
    ``peak`` decide working vs. silent. When permission is ``authorized`` a silent
    capture is clearly a device/level issue, not a permission one — the message says so.
    """
    if permission == "denied":
        return MicTestResult(
            ok=False,
            verdict="denied",
            message=(
                "Microphone permission is DENIED for this app — enable your terminal/app in "
                "System Settings › Privacy & Security › Microphone, then quit & reopen it."
            ),
            permission=permission,
        )
    if permission == "restricted":
        return MicTestResult(
            ok=False,
            verdict="restricted",
            message="Microphone access is restricted by a system policy (e.g. MDM / Screen Time).",
            permission=permission,
        )
    if error is not None:
        return MicTestResult(ok=False, verdict="error", message=error, permission=permission)
    if not captured:
        return MicTestResult(
            ok=False,
            verdict="no_device",
            message="No microphone available — connect an input device and grant mic access.",
            permission=permission,
        )
    level = int(round(min(1.0, max(0.0, peak)) * 100))
    if peak < _SILENCE_PEAK:
        if permission == "notDetermined":
            message = (
                "No audio yet — macOS hasn't granted mic access; it should prompt on the first "
                f"capture. If no prompt appears, {_MIC_PERMISSION_HINT}."
            )
        elif permission == "authorized":
            message = (
                "Permission is granted but no audio — check the selected input device and that "
                "the microphone isn't muted."
            )
        else:
            message = f"No audio — {_MIC_PERMISSION_HINT}."
        return MicTestResult(
            ok=False,
            verdict="silent",
            message=message,
            level=level,
            rms=rms,
            peak=peak,
            permission=permission,
        )
    return MicTestResult(
        ok=True,
        verdict="ok",
        message=f"Microphone OK — level {level}%",
        level=level,
        rms=rms,
        peak=peak,
        permission=permission,
    )


def mic_test(sample_rate: int, *, seconds: float = 1.5, frame_samples: int = 1280) -> MicTestResult:
    """Capture ~``seconds`` from the input device and report a clear verdict.

    Never raises: a missing ``sounddevice`` / PortAudio / input device, or any
    capture error, is turned into an ``error`` / ``no_device`` verdict so the
    server stays up. Computes peak + RMS over the captured float32 mono frames and
    delegates the working / silent / error decision to :func:`mic_test_verdict`.
    On macOS a silent (near-zero) capture is the tell-tale sign of an ungranted
    microphone permission — the verdict says exactly that.
    """
    permission = mic_permission_status()
    try:
        sd = _sd()
    except Exception as exc:  # noqa: BLE001 — no audio extra / no PortAudio
        return mic_test_verdict(
            captured=False,
            rms=0.0,
            peak=0.0,
            error=f"audio capture unavailable: {exc}",
            permission=permission,
        )
    if not mic_available():
        return mic_test_verdict(captured=False, rms=0.0, peak=0.0, permission=permission)

    frames_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames_q.put(indata[:, 0].copy())

    collected: list[np.ndarray] = []
    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            callback=_callback,
        ):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                try:
                    collected.append(frames_q.get(timeout=0.1))
                except queue.Empty:
                    continue
    except Exception as exc:  # noqa: BLE001 — device opened/failed mid-capture
        return mic_test_verdict(
            captured=False,
            rms=0.0,
            peak=0.0,
            error=f"microphone error: {exc}",
            permission=permission,
        )

    # Drain any frames the callback queued right at/after the deadline (and, for a
    # synchronous-callback backend, every frame) so the last syllable isn't lost.
    while True:
        try:
            collected.append(frames_q.get_nowait())
        except queue.Empty:
            break

    if not collected:
        return mic_test_verdict(
            captured=False,
            rms=0.0,
            peak=0.0,
            error="no audio frames captured",
            permission=permission,
        )
    samples = np.concatenate(collected)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    return mic_test_verdict(captured=True, rms=rms, peak=peak, permission=permission)


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
    wake: Any,
    sample_rate: int,
    *,
    frame_samples: int = 1280,
    poll_seconds: float = 0.1,
    stop: threading.Event | None = None,
) -> bool:
    """Block until ``wake.detect(frame)`` fires on an 80 ms (1280-sample) frame.

    Returns ``True`` when the wake word fired, or ``False`` if ``stop`` was set
    before it fired (so a GUI-driven wake loop can be torn down between frames
    while it sits idle waiting for the phrase).
    """
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
            if stop is not None and stop.is_set():
                return False
            try:
                frame = frames_q.get(timeout=poll_seconds)
            except queue.Empty:
                continue
            if wake.detect(frame):
                return True
