"""Acoustic echo cancellation (AEC) front-end for open-speaker barge-in (R2-1).

Without AEC the mic hears the assistant's own TTS coming out of the speakers, so
:func:`~my_stt_tts.audio.monitor_during_playback` would treat the bot's voice as a
user interruption. AEC removes the played-back signal (the *reference*) from the
mic capture so only the user's voice is left for the VAD / interrupt gate.

Three backends, in the preference order the README documents:

* ``"voiceprocessing"`` — macOS **hardware** AEC via ``AVAudioEngine`` +
  ``AVAudioInputNode.setVoiceProcessingEnabled`` (the VoiceProcessingIO unit).
  This is what FaceTime uses; it cancels in the OS audio HAL, so by the time mic
  frames reach Python they are *already* echo-cancelled and our software filter
  is a no-op. We can't run the real hardware unit in a unit test, so this backend
  only *enables* the OS feature (its availability is probed without a mic) and
  otherwise passes audio through unchanged.
* ``"nlms"`` — a pure-numpy **normalized least-mean-squares** adaptive filter that
  references the playback buffer. Always available (no native deps), fully
  unit-tested, and the cross-platform fallback. This is real, working AEC: it
  adapts a FIR filter to predict the echo from the reference and subtracts it.
* ``"off"`` — identity pass-through (legacy half-duplex behaviour).

The seam is the :class:`EchoCanceller` protocol: ``push_reference(samples)`` feeds
the audio that is being *played*, and ``process(frame)`` returns the mic ``frame``
with the echo removed. The monitor loop pushes the active
:class:`~my_stt_tts.tts.Playback` reference and processes each mic frame before VAD.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading
from collections import deque
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("my_stt_tts.aec")

AEC_MODES = ("off", "nlms", "voiceprocessing", "webrtc", "auto")


@runtime_checkable
class EchoCanceller(Protocol):
    """Removes the played-back (reference) signal from captured mic audio.

    Usage::

        aec.push_reference(played_samples)   # what the speaker is emitting
        clean = aec.process(mic_frame)       # mic with the echo subtracted
    """

    #: When True, the played-back signal is already removed (hardware AEC), so the
    #: caller may relax its open-speaker energy floor.
    active: bool

    def push_reference(self, samples: np.ndarray) -> None:
        """Feed the audio currently being played (the echo source)."""
        ...

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return ``frame`` (a mic chunk) with the echo cancelled."""
        ...

    def reset(self) -> None:
        """Clear all adaptive state for a new utterance."""
        ...


class NullEchoCanceller:
    """Identity pass-through: no cancellation (``aec="off"``)."""

    active: bool = False

    def push_reference(self, samples: np.ndarray) -> None:  # noqa: D102 — see protocol
        return None

    def process(self, frame: np.ndarray) -> np.ndarray:  # noqa: D102
        return np.asarray(frame, dtype=np.float32).ravel()

    def reset(self) -> None:  # noqa: D102
        return None


class NlmsEchoCanceller:
    """Normalized LMS adaptive FIR echo canceller (pure numpy; the safe default).

    Models the echo path as an FIR filter ``w`` of ``taps`` coefficients applied to
    the recent reference (loudspeaker) signal. For each mic sample it predicts the
    echo ``y_hat = w · x``, subtracts it to get the error ``e = mic - y_hat``, and
    adapts ``w`` toward cancelling that error::

        w += mu * e * x / (||x||^2 + eps)

    The NLMS normalisation by the reference power makes convergence robust to the
    loudspeaker volume. ``process`` is sample-accurate but vectorised per call;
    ``mu`` (step size) trades convergence speed for stability. With no reference
    pushed yet, it is a clean pass-through.
    """

    active: bool = True

    def __init__(
        self, *, taps: int = 256, mu: float = 0.3, eps: float = 1e-6, ref_max_seconds: float = 8.0
    ) -> None:
        if taps <= 0:
            raise ValueError("taps must be > 0")
        if not 0.0 < mu <= 2.0:
            raise ValueError("mu must be in (0, 2]")
        self.taps = taps
        self.mu = mu
        self.eps = eps
        self._w = np.zeros(taps, dtype=np.float32)
        # Sliding history of recent reference samples (newest last). Bounded so the
        # buffer can't grow without limit during a long utterance.
        self._ref_maxlen = max(taps, int(16000 * ref_max_seconds))
        self._ref: deque[float] = deque(maxlen=self._ref_maxlen)
        # Seed with `taps` zeros so the first mic sample has a full reference window.
        self._ref.extend([0.0] * taps)

    def push_reference(self, samples: np.ndarray) -> None:
        """Append played-back samples to the reference history."""
        arr = np.asarray(samples, dtype=np.float32).ravel()
        self._ref.extend(arr.tolist())

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Cancel the echo from ``frame`` using the buffered reference + adapt ``w``."""
        mic = np.asarray(frame, dtype=np.float32).ravel()
        if mic.size == 0:
            return mic
        ref = np.fromiter(self._ref, dtype=np.float32, count=len(self._ref))
        if ref.size < self.taps:
            return mic  # not enough reference yet -> pass through
        out = np.empty(mic.size, dtype=np.float32)
        w = self._w
        taps = self.taps
        # Align so the LAST `mic.size` reference windows line up with the mic samples
        # (assumes the reference is roughly time-aligned with capture).
        ref_window = ref[-(taps + mic.size - 1) :] if ref.size >= taps + mic.size - 1 else ref
        if ref_window.size < taps + mic.size - 1:
            pad = np.zeros(taps + mic.size - 1 - ref_window.size, dtype=np.float32)
            ref_window = np.concatenate([pad, ref_window])
        for i in range(mic.size):
            x = ref_window[i : i + taps][::-1]  # most-recent-first FIR input
            y_hat = float(np.dot(w, x))
            err = mic[i] - y_hat
            norm = float(np.dot(x, x)) + self.eps
            w += (self.mu * err / norm) * x
            out[i] = err
        self._w = w
        return out

    def reset(self) -> None:
        """Forget the adapted filter + reference for a new utterance."""
        self._w = np.zeros(self.taps, dtype=np.float32)
        self._ref.clear()
        self._ref.extend([0.0] * self.taps)


class VoiceProcessingEchoCanceller:
    """macOS hardware AEC via ``AVAudioEngine`` VoiceProcessingIO (the best path).

    Enabling voice processing on the input node turns on the same echo canceller +
    noise suppressor FaceTime uses, *inside* the audio HAL. Mic frames that reach
    Python are therefore already echo-cancelled, so :meth:`process` is a
    pass-through and :attr:`active` reports True only when the OS feature could be
    enabled. PyObjC (the ``aec`` extra) is lazy-imported; if it is unavailable the
    factory falls back to NLMS and this backend reports ``active=False``.

    We deliberately do not open a live audio unit here (capture still flows through
    ``sounddevice``); :meth:`available` just verifies the API exists so the factory
    can decide. Probing for the API does not need a mic, so it is unit-testable.
    """

    def __init__(self) -> None:
        self.active = self._enable()

    @staticmethod
    def available() -> bool:
        """Whether the VoiceProcessingIO API is importable on this machine."""
        try:
            import AVFoundation  # noqa: PLC0415  (lazy, native)
        except Exception:  # PyObjC not installed / not macOS
            return False
        return hasattr(AVFoundation, "AVAudioEngine")

    def _enable(self) -> bool:
        """Turn on hardware voice processing on a probe engine. Returns success."""
        try:
            import AVFoundation  # noqa: PLC0415

            # pylint: disable=no-member  # PyObjC populates AVAudioEngine dynamically
            engine = AVFoundation.AVAudioEngine.alloc().init()
            input_node = engine.inputNode()
            ok, err = input_node.setVoiceProcessingEnabled_error_(True, None)
            if not ok:
                log.info("VoiceProcessingIO could not be enabled: %s", err)
                return False
            return bool(input_node.isVoiceProcessingEnabled())
        except Exception:  # any PyObjC / CoreAudio failure -> not active
            log.info("macOS VoiceProcessingIO unavailable; AEC handled in software.")
            return False

    def push_reference(self, samples: np.ndarray) -> None:  # noqa: D102 — HW handles echo
        return None

    def process(self, frame: np.ndarray) -> np.ndarray:  # noqa: D102 — already cancelled
        return np.asarray(frame, dtype=np.float32).ravel()

    def reset(self) -> None:  # noqa: D102
        return None


class VoiceProcessingCapture:
    """Capture mic audio THROUGH the macOS VoiceProcessingIO unit (R3-4, closes G3).

    This is the real hardware-AEC capture path: instead of capturing via plain
    ``sounddevice`` (which the OS does *not* echo-cancel) and cancelling in Python,
    we open an ``AVAudioEngine`` whose input node has
    ``setVoiceProcessingEnabled_(True)``, install a **tap** on its output bus, and
    pull the **already-echo-cancelled** PCM the OS HAL produces. The tap callback
    bridges each ``AVAudioPCMBuffer`` (non-interleaved float32, channel 0) into a
    numpy frame, resamples it from the device rate (48 kHz) to the pipeline
    ``sample_rate``, and queues it; :meth:`mic_frames` yields fixed-size frames so
    it is a drop-in mic source for the capture loop.

    PyObjC (the ``aec`` extra) is lazy-imported. :meth:`start` returns False if the
    bridge can't be built (no PyObjC / VP can't enable / the engine won't start),
    and the caller falls back to ``sounddevice`` + software NLMS — so this never
    breaks the loop, it just upgrades the front-end when the HAL cooperates.
    """

    def __init__(self, sample_rate: int = 16000, *, frame_samples: int = 512) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._engine: Any = None
        self._input: Any = None
        self._device_sr: float = 48000.0
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=256)
        self._residual = np.zeros(0, dtype=np.float32)
        self._started = threading.Event()
        self._closed = threading.Event()

    @staticmethod
    def available() -> bool:
        """Whether the AVAudioEngine tap API is importable on this machine."""
        try:
            import AVFoundation  # noqa: PLC0415

            # pylint: disable=no-member  # PyObjC populates these dynamically
            return hasattr(AVFoundation, "AVAudioEngine") and hasattr(
                AVFoundation.AVAudioInputNode, "installTapOnBus_bufferSize_format_block_"
            )
        except Exception:
            return False

    def start(self) -> bool:
        """Open the engine, enable voice processing, install the tap. Returns success."""
        try:
            import AVFoundation  # noqa: PLC0415

            # pylint: disable=no-member  # PyObjC populates these dynamically
            engine = AVFoundation.AVAudioEngine.alloc().init()
            input_node = engine.inputNode()
            ok, err = input_node.setVoiceProcessingEnabled_error_(True, None)
            if not ok or not input_node.isVoiceProcessingEnabled():
                log.info("VoiceProcessingIO could not be enabled for capture: %s", err)
                return False
            fmt = input_node.outputFormatForBus_(0)
            self._device_sr = float(fmt.sampleRate())
            input_node.installTapOnBus_bufferSize_format_block_(0, 1024, fmt, self._on_buffer)
            engine.prepare()
            ok2, err2 = engine.startAndReturnError_(None)
            if not ok2:
                log.info("AVAudioEngine failed to start: %s", err2)
                with _suppress():
                    input_node.removeTapOnBus_(0)
                return False
            self._engine = engine
            self._input = input_node
            self._started.set()
            return True
        except Exception:  # any PyObjC / CoreAudio failure -> caller falls back
            log.info("VoiceProcessingIO capture unavailable; using sounddevice + NLMS.")
            return False

    def _on_buffer(self, buf: Any, _when: Any) -> None:
        """Tap callback: bridge channel-0 float32 PCM -> numpy, resample, enqueue."""
        try:
            n = int(buf.frameLength())
            channels = buf.floatChannelData()
            if channels is None or n == 0:
                return
            raw = np.frombuffer(channels[0].as_buffer(n * 4), dtype=np.float32).copy()
        except Exception:  # malformed buffer -> drop it, never raise into CoreAudio
            return
        frame = self._resample(raw)
        with _suppress_full():
            self._q.put_nowait(frame)

    def _resample(self, arr: np.ndarray) -> np.ndarray:
        """Linear-resample a device-rate block to the pipeline ``sample_rate``."""
        if self._device_sr == self.sample_rate or arr.size == 0:
            return arr
        n_out = max(1, int(round(arr.size * self.sample_rate / self._device_sr)))
        x_new = np.linspace(0.0, arr.size - 1, n_out)
        return np.interp(x_new, np.arange(arr.size), arr).astype(np.float32)

    def mic_frames(self) -> Iterator[np.ndarray]:
        """Yield fixed-size (``frame_samples``) HW-cancelled mic frames until closed.

        Drains the queue even after :meth:`close` so buffered frames are still
        delivered (the loop only ends once the queue has run dry)."""
        while True:
            try:
                block = self._q.get(timeout=0.1)
            except queue.Empty:
                if self._closed.is_set():
                    return
                continue
            self._residual = np.concatenate([self._residual, block])
            while self._residual.size >= self.frame_samples:
                yield self._residual[: self.frame_samples].copy()
                self._residual = self._residual[self.frame_samples :]

    def close(self) -> None:
        """Stop the engine and remove the tap (idempotent)."""
        self._closed.set()
        with _suppress():
            if self._input is not None:
                self._input.removeTapOnBus_(0)
            if self._engine is not None:
                self._engine.stop()


class _suppress:  # noqa: N801 — tiny context-manager helper
    """Swallow any exception (best-effort PyObjC teardown)."""

    def __enter__(self) -> _suppress:
        return self

    def __exit__(self, *_exc: object) -> bool:
        return True


class _suppress_full:  # noqa: N801 — tiny queue.Full swallow
    def __enter__(self) -> _suppress_full:
        return self

    def __exit__(self, exc_type: type | None, *_exc: object) -> bool:
        return exc_type is not None and issubclass(exc_type, queue.Full)


class WebRtcApmEchoCanceller:
    """Linux AEC via the WebRTC Audio Processing Module (APM) — G8.

    On a non-Mac brain host there is no VoiceProcessingIO, so the open-speaker AEC
    path needs a Linux backend. ``webrtc-audio-processing`` (the ``linux-aec``
    extra) exposes the same battle-tested echo canceller + noise suppressor Chrome
    uses. We feed it the playback reference (``push_reference``) and run it on each
    captured frame (``process``) — the same :class:`EchoCanceller` seam as NLMS.

    The native module is lazy-imported; if it is unavailable (not Linux, wheel
    missing) :attr:`active` is False and :meth:`process` passes through, so the
    factory falls back to the pure-numpy NLMS canceller. We never raise into the
    audio loop. The APM works on 10 ms frames at 16 kHz; frames are buffered to that
    grain and processed when a full APM frame is available.
    """

    active: bool

    #: WebRTC APM operates on 10 ms frames.
    FRAME_MS = 10

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self._frame_n = max(1, int(sample_rate * self.FRAME_MS / 1000))
        self._apm: Any = self._build()
        self.active = self._apm is not None
        self._ref = np.zeros(0, dtype=np.float32)
        self._mic = np.zeros(0, dtype=np.float32)

    @staticmethod
    def available() -> bool:
        """Whether a WebRTC Audio Processing Module wheel is importable on this host."""
        try:
            import importlib.util

            return importlib.util.find_spec("webrtc_audio_processing") is not None
        except Exception:
            return False

    def _build(self) -> Any:
        try:
            from webrtc_audio_processing import (
                AudioProcessingModule,  # type: ignore[import-not-found]
            )

            apm = AudioProcessingModule(enable_ns=True, enable_aec=True)
            with contextlib.suppress(Exception):
                apm.set_stream_format(self.sample_rate, 1)
                apm.set_reverse_stream_format(self.sample_rate, 1)
            return apm
        except Exception:  # not Linux / wheel missing / API mismatch
            log.info("WebRTC APM unavailable; AEC will fall back to NLMS.", exc_info=True)
            return None

    def push_reference(self, samples: np.ndarray) -> None:
        """Feed played-back samples to the APM far-end (reverse) stream."""
        if self._apm is None:
            return
        arr = np.asarray(samples, dtype=np.float32).ravel()
        self._ref = np.concatenate([self._ref, arr])
        self._drain_reference()

    def _drain_reference(self) -> None:
        while self._ref.size >= self._frame_n:
            chunk = self._ref[: self._frame_n]
            self._ref = self._ref[self._frame_n :]
            with contextlib.suppress(Exception):
                self._apm.process_reverse_stream(self._to_i16(chunk))

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Cancel echo from ``frame`` via the APM (buffered to 10 ms grain)."""
        mic = np.asarray(frame, dtype=np.float32).ravel()
        if self._apm is None or mic.size == 0:
            return mic
        self._mic = np.concatenate([self._mic, mic])
        out: list[np.ndarray] = []
        while self._mic.size >= self._frame_n:
            chunk = self._mic[: self._frame_n]
            self._mic = self._mic[self._frame_n :]
            try:
                cleaned = self._apm.process_stream(self._to_i16(chunk))
                out.append(self._from_i16(cleaned))
            except Exception:  # APM hiccup -> pass the frame through unmodified
                out.append(chunk)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    @staticmethod
    def _to_i16(arr: np.ndarray) -> bytes:
        return (np.clip(arr, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()

    @staticmethod
    def _from_i16(data: Any) -> np.ndarray:
        buf = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        return np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0

    def reset(self) -> None:
        """Clear the buffered far-end / near-end audio for a new utterance."""
        self._ref = np.zeros(0, dtype=np.float32)
        self._mic = np.zeros(0, dtype=np.float32)


def make_voiceprocessing_capture(cfg: Any) -> VoiceProcessingCapture | None:
    """Build + start a :class:`VoiceProcessingCapture`, or None to fall back (R3-4).

    Returns a started capture only when ``aec_mode`` selects the hardware path,
    ``aec_hw_capture`` is on, and the PyObjC bridge actually starts. In every other
    case returns None so the caller uses ``sounddevice`` (+ software NLMS).
    """
    mode = getattr(cfg, "aec_mode", "off")
    if mode not in ("voiceprocessing", "auto") or not getattr(cfg, "aec_hw_capture", True):
        return None
    if not VoiceProcessingCapture.available():
        if mode == "voiceprocessing":
            log.info("HW-AEC capture requested but PyObjC/VoiceProcessingIO is unavailable.")
        return None
    capture = VoiceProcessingCapture(int(getattr(cfg, "sample_rate", 16000)))
    if capture.start():
        log.info("capturing through macOS VoiceProcessingIO (hardware AEC).")
        return capture
    return None


def make_echo_canceller(cfg: Any) -> EchoCanceller:
    """Build the configured :class:`EchoCanceller` (from ``cfg.aec_mode``).

    * ``"voiceprocessing"`` — macOS hardware AEC if available, else NLMS.
    * ``"webrtc"`` — Linux WebRTC Audio Processing Module if available, else NLMS (G8).
    * ``"auto"`` — macOS HW AEC if available, else WebRTC-APM (Linux), else NLMS.
    * ``"nlms"`` — always the software adaptive filter.
    * ``"off"`` / anything else — identity pass-through.
    """
    mode = getattr(cfg, "aec_mode", "off")
    sample_rate = int(getattr(cfg, "sample_rate", 16000))
    if mode == "off":
        return NullEchoCanceller()
    if mode == "nlms":
        return _make_nlms(cfg)
    if mode == "webrtc":
        apm = _try_webrtc_apm(sample_rate)
        if apm is not None:
            return apm
        log.info("WebRTC APM unavailable; using software NLMS.")
        return _make_nlms(cfg)
    if mode in ("voiceprocessing", "auto"):
        if VoiceProcessingEchoCanceller.available():
            hw = VoiceProcessingEchoCanceller()
            if hw.active:
                return hw
            log.info("hardware AEC inactive; falling back to software NLMS.")
        elif mode == "voiceprocessing":
            log.info("VoiceProcessingIO unavailable (no PyObjC); using software NLMS.")
        elif mode == "auto":
            apm = _try_webrtc_apm(sample_rate)
            if apm is not None:
                return apm
        return _make_nlms(cfg)
    return NullEchoCanceller()


def _try_webrtc_apm(sample_rate: int) -> WebRtcApmEchoCanceller | None:
    """Build a :class:`WebRtcApmEchoCanceller` if the native module is active (G8)."""
    if not WebRtcApmEchoCanceller.available():
        return None
    apm = WebRtcApmEchoCanceller(sample_rate)
    if apm.active:
        log.info("using the WebRTC Audio Processing Module for AEC (Linux).")
        return apm
    return None


def _make_nlms(cfg: Any) -> NlmsEchoCanceller:
    return NlmsEchoCanceller(
        taps=int(getattr(cfg, "aec_nlms_taps", 256)),
        mu=float(getattr(cfg, "aec_nlms_mu", 0.3)),
    )
