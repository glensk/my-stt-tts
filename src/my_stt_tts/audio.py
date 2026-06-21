"""Audio capture, pre-roll ring buffer, playback, half-duplex mic gating, and
wake-word / VAD-gated capture helpers.

``sounddevice`` is lazy-imported (needs PortAudio, from the ``audio`` extra); the
buffer and gate logic are pure and unit-tested. The streaming helpers
(``record_*`` and ``listen_for_wake``) need a real mic, so they are not unit-tested.
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
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


def resample_to(arr: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear-resample float32 mono PCM from ``src_rate`` to ``dst_rate`` (pure).

    Identity when the rates already match (or the input is empty). Dependency-free
    (``np.interp``), which is good enough for VAD/STT framing — the wake word and
    parakeet/Silero models expect **16 kHz mono**, and a 48 kHz device stream fed to
    them un-resampled is heard as garbage (the wake never fires, STT returns
    nothing). Used to guarantee the audio reaching the models is truly the pipeline
    rate regardless of what the input device delivered.
    """
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if src_rate == dst_rate or arr.size == 0:
        return arr
    n_out = max(1, int(round(arr.size * dst_rate / src_rate)))
    x_new = np.linspace(0.0, arr.size - 1, n_out)
    return np.interp(x_new, np.arange(arr.size), arr).astype(np.float32)


def reframe(arr: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    """Split a 1-D float32 array into fixed-size frames, zero-padding the last one.

    Silero VAD accepts **only** an exact chunk size (512 samples at 16 kHz); a
    differently-sized frame raises inside the TorchScript model. The capture
    callbacks deliver device-blocksize chunks (often 1024/2048/4096), so this
    re-chunks a captured block into model-sized frames before VAD. The trailing
    partial frame is right-padded with zeros so no audio is silently dropped.
    """
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if frame_samples <= 0 or arr.size == 0:
        return []
    out: list[np.ndarray] = []
    for start in range(0, arr.size, frame_samples):
        chunk = arr[start : start + frame_samples]
        if chunk.size < frame_samples:
            chunk = np.concatenate([chunk, np.zeros(frame_samples - chunk.size, dtype=np.float32)])
        out.append(chunk.astype(np.float32, copy=False))
    return out


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


def mic_permission_status(cfg: Any = None) -> str:
    """Microphone permission status for this app, WITHOUT capturing audio (per-OS).

    Delegates to :func:`my_stt_tts.platform.mic_permission_status`, which branches
    per OS: macOS returns the real TCC grant (``authorized`` / ``denied`` /
    ``notDetermined`` / ``restricted``); Linux has no per-app permission model so it
    reports ``n/a`` when a capture device exists (``unavailable`` when none does);
    Windows reads the system microphone privacy toggle (``authorized`` / ``denied``)
    or falls back to a device check. Never raises — failures read as ``unavailable``
    so callers fall back to the capture-based verdict. ``cfg`` (optional) lets a
    test / cross-host setup pin the platform.
    """
    from .platform import mic_permission_status as _status

    return _status(cfg)


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
        elif permission in ("authorized", "n/a"):
            # Either macOS granted it, or this OS has no per-app permission gate
            # (Linux/Windows device present) — so a silent capture is a device/level
            # issue, not a permission one.
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


def record_fixed(sample_rate: int, *, seconds: float = 3.0) -> tuple[np.ndarray, int]:
    """Capture a fixed ``seconds`` of mic audio at the device rate (float32 mono).

    Returns ``(clip, device_rate)`` where ``clip`` is the RAW capture at the rate the
    device actually opened (``device_rate`` — commonly 48 kHz when 16 kHz isn't
    honoured, see :func:`_supported_capture_rate`) and is **NOT** resampled. This is
    the human "record & replay" path: the caller must play ``clip`` back at the SAME
    ``device_rate`` it was captured at, or the replay is sped up / high-pitched
    (a 16 kHz clip played at 24/48 kHz plays 1.5×/3× too fast). Resampling to the
    16 kHz pipeline rate is only for STT/wake — never for this faithful round-trip.
    ``sample_rate`` is the *requested* rate (the device-rate probe starts from it).
    Unlike :func:`record_until_silence` there is no VAD/endpointing — it records the
    full window, so the replay plays back exactly what the mic heard. Raises on a
    device/PortAudio failure; the caller (a worker thread) is expected to guard it.
    """
    sd = _sd()
    device_rate = _supported_capture_rate(sd, sample_rate)
    frames_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames_q.put(indata[:, 0].copy())

    collected: list[np.ndarray] = []
    with sd.InputStream(
        samplerate=device_rate,
        channels=1,
        dtype="float32",
        blocksize=1280,
        callback=_callback,
    ):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                collected.append(frames_q.get(timeout=0.1))
            except queue.Empty:
                continue
    # Drain anything queued right at the deadline so the tail isn't lost.
    while True:
        try:
            collected.append(frames_q.get_nowait())
        except queue.Empty:
            break
    raw = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
    # Return the RAW capture at the device rate — the replay plays it at this same
    # rate for faithful pitch/speed. (Do NOT resample to 16 kHz here.)
    return raw, device_rate


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


def _supported_capture_rate(sd: Any, requested: int) -> int:
    """Return a samplerate the input device will actually open at (≈ ``requested``).

    PortAudio resamples to a requested rate on most backends, but not all — on a
    host where 16 kHz isn't natively honoured the device opens at its native rate
    (commonly 48 kHz) and the model gets garbage. This probes ``check_input_settings``
    for the requested rate and, if that fails, returns the device's
    ``default_samplerate`` so the caller can capture at the native rate and resample
    in Python (guaranteeing true 16 kHz reaches the wake/VAD/STT models). Best-effort
    — falls back to ``requested`` if the probe itself errors.
    """
    try:
        sd.check_input_settings(samplerate=requested, channels=1, dtype="float32")
        return requested
    except Exception:  # noqa: BLE001 — device won't open at the requested rate
        try:
            dev = sd.query_devices(kind="input")
            native = int(round(float(dev.get("default_samplerate", requested))))
            return native or requested
        except Exception:  # noqa: BLE001 — no device info -> just try the requested rate
            return requested


# The ±full-scale of the int16 PCM the wake model actually consumes. The float32
# pipeline carries [-1, 1]; openWakeWord's AudioFeatures re-casts to int16, so the
# magnitude it truly sees is the float amplitude × 32767. The GUI classifies the
# capture by this int16 magnitude, NOT the abstract 0..1 float (which hides "is
# this loud enough for the model to score?" — the PRIME wake-failure suspect).
_INT16_FULL_SCALE = 32767.0
# int16-peak thresholds for ``expected_level``: a near-silent built-in mic peaks
# at a few hundred (the model sees near-zero mels → score collapses); a healthy
# capture sits in the low thousands; anything brushing full-scale is clipping.
_INT16_LOW_PEAK = 2000  # below this the capture is too quiet for the wake model
_INT16_HIGH_PEAK = 30000  # above this the capture is hot / clipping (±32768 wraps)


def _expected_level(int16_peak: int) -> str:
    """Classify an int16 peak magnitude as ``low`` / ``ok`` / ``high`` (pure).

    The single most useful wake-failure tell: a ``low`` capture (int16 peak under
    :data:`_INT16_LOW_PEAK`) starves openWakeWord's mel features and pins the score
    near zero regardless of the word — exactly the dead-wake symptom. ``high`` flags
    a clipping/overloaded capture; ``ok`` is the usable band in between.
    """
    if int16_peak < _INT16_LOW_PEAK:
        return "low"
    if int16_peak > _INT16_HIGH_PEAK:
        return "high"
    return "ok"


def _snr_db(arr: np.ndarray, sample_rate: int, *, vad: Any = None) -> float | None:
    """Estimate speech-vs-noise-floor SNR in dB over short windows (None if N/A).

    Splits the clip into 20 ms windows and computes each window's RMS. The noise
    floor is the quietest 10% of windows (or, when a Silero ``vad`` is supplied, the
    NON-speech windows); the signal is the loudest. ``SNR = 20·log10(sig / noise)``.
    Returns ``None`` when the clip is too short to have ≥2 windows or the noise floor
    is exactly zero (digital silence — an undefined ratio). Pure + dependency-free
    in the quietest-10% path; the VAD path is used only when an extra is installed.
    """
    if arr.size == 0 or sample_rate <= 0:
        return None
    win = max(1, int(0.02 * sample_rate))  # 20 ms windows
    n_win = arr.size // win
    if n_win < 2:
        return None
    trimmed = arr[: n_win * win].reshape(n_win, win)
    win_rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
    speech_mask: np.ndarray | None = None
    if vad is not None:
        with contextlib.suppress(Exception):
            flags = [bool(vad.is_speech(trimmed[i])) for i in range(n_win)]
            mask = np.asarray(flags, dtype=bool)
            if mask.any() and (~mask).any():
                speech_mask = mask
    if speech_mask is not None:
        noise = float(np.median(win_rms[~speech_mask]))
        signal = float(np.median(win_rms[speech_mask]))
    else:
        order = np.sort(win_rms)
        k = max(1, n_win // 10)  # quietest 10% as the noise floor
        noise = float(np.mean(order[:k]))
        signal = float(np.mean(order[-k:]))
    if noise <= 0.0 or signal <= 0.0:
        return None
    return round(20.0 * float(np.log10(signal / noise)), 2)


def _lufs(arr: np.ndarray, sample_rate: int) -> float | None:
    """Integrated loudness (LUFS) via ``pyloudnorm`` — ``None`` if unimportable.

    ``pyloudnorm`` lives in the optional ``debug`` extra so the core package stays
    clean: when it (or its scipy dependency) is missing, or the clip is too short
    for the meter's 400 ms block, this returns ``None`` and the field is simply
    absent-as-null. Best-effort — never raises into a diagnostic.
    """
    if arr.size == 0 or sample_rate <= 0:
        return None
    try:
        import pyloudnorm as pyln
    except Exception:  # noqa: BLE001 — the `debug` extra isn't installed; degrade to None
        return None
    if arr.size < sample_rate * 0.4:  # the BS.1770 meter needs a ≥400 ms block
        return None
    try:
        meter = pyln.Meter(sample_rate)
        value = float(meter.integrated_loudness(arr.astype(np.float64)))
    except Exception:  # noqa: BLE001 — silent/degenerate input -> undefined loudness
        return None
    if not np.isfinite(value):  # pyloudnorm returns -inf for digital silence
        return None
    return round(value, 2)


def _true_peak_db(arr: np.ndarray) -> float:
    """Inter-sample (true) peak in dBFS via 4× oversampling (scipy when available).

    A signal can peak BETWEEN samples above its sample peak (inter-sample / true
    peak), which is what a DAC reconstructs and can clip. Oversamples 4× with
    ``scipy.signal.resample_poly`` (a band-limited polyphase resampler), then takes
    the peak. ``scipy`` is only pulled in by extras, so without it this degrades to
    the plain sample peak (still correct, just not inter-sample). dBFS relative to
    full scale 1.0; ``-inf``-ish silence is floored to a large negative dB.
    """
    if arr.size == 0:
        return -120.0
    peak = float(np.max(np.abs(arr)))
    with contextlib.suppress(Exception):
        from scipy.signal import resample_poly  # type: ignore[import-untyped]

        upsampled = resample_poly(arr.astype(np.float64), 4, 1)
        peak = max(peak, float(np.max(np.abs(upsampled))))
    if peak <= 0.0:
        return -120.0
    return round(20.0 * float(np.log10(min(peak, 1.0) if peak <= 1.0 else peak)), 2)


def capture_stats(samples: np.ndarray, sample_rate: int, *, vad: Any = None) -> dict[str, Any]:
    """Summarise a captured clip for the debug instrument and the wake diagnostics.

    The base fields (``sample_rate`` / ``samples`` / ``duration_s`` / ``rms`` /
    ``peak``) are the original capture summary. The extended fields make the PRIME
    wake-failure suspect — a too-quiet capture — instantly visible:

    * ``int16_peak`` / ``int16_rms`` — the magnitude on the ±32768 int16 scale the
      wake model ACTUALLY sees (float amplitude × 32767). A built-in mic peaking at
      0.05 float = ~1638 int16 → starved mel features → score collapses.
    * ``expected_level`` — ``low`` / ``ok`` / ``high`` classification of ``int16_peak``
      (see :func:`_expected_level`): the at-a-glance "is the capture loud enough?".
    * ``crest_db`` — crest factor ``20·log10(peak / rms)``: speech runs ~12–20 dB; a
      near-flat value hints at a DC-y / clipped / noise-only capture.
    * ``dc_offset`` — the signal mean (a non-zero bias starves the model + biases VAD).
    * ``true_peak_db`` — inter-sample true peak in dBFS (4× oversampled; see
      :func:`_true_peak_db`).
    * ``snr_db`` — speech-vs-noise-floor SNR in dB (None when indeterminable); a
      Silero ``vad`` (the ``vad`` extra) gates speech windows, else quietest-10%.
    * ``lufs`` — integrated loudness via the optional ``debug`` extra's ``pyloudnorm``
      (None when the extra is absent), so core stays clean.
    """
    arr = np.asarray(samples, dtype=np.float32).ravel()
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
    int16_peak = int(round(min(peak, 1.0) * _INT16_FULL_SCALE))
    int16_rms = int(round(min(rms, 1.0) * _INT16_FULL_SCALE))
    dc_offset = float(np.mean(arr)) if arr.size else 0.0
    crest_db = round(20.0 * float(np.log10(peak / rms)), 2) if rms > 0.0 and peak > 0.0 else 0.0
    return {
        "sample_rate": int(sample_rate),
        "samples": int(arr.size),
        "duration_s": round(arr.size / sample_rate, 3) if sample_rate else 0.0,
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "int16_peak": int16_peak,
        "int16_rms": int16_rms,
        "expected_level": _expected_level(int16_peak),
        "crest_db": crest_db,
        "dc_offset": round(dc_offset, 6),
        "true_peak_db": _true_peak_db(arr),
        "snr_db": _snr_db(arr, int(sample_rate), vad=vad),
        "lufs": _lufs(arr, int(sample_rate)),
    }


# --------------------------------------------------------------------------- #
# Mic-diagnostic recordings: software gain, level-over-time, on-disk archive   #
# --------------------------------------------------------------------------- #

# Number of evenly-spaced per-window peak magnitudes the GUI plots as a
# level-over-time graph. ~48 windows is a smooth-enough envelope for a ~2 s clip
# without flooding the result event with samples.
_LEVELS_WINDOWS = 48


def apply_gain(clip: np.ndarray, gain: float) -> np.ndarray:
    """Apply a software input ``gain`` to a float32 mono clip, clip-protected.

    Multiplies by ``gain`` and hard-clips to ±1.0 so a hot capture is never folded
    into garbage by overflow — a quiet server mic can be lifted to a usable level
    without the wake/STT models seeing wrapped samples. ``gain`` of 1.0 (or an empty
    clip) is an identity pass. Pure: returns a new array, leaves the input untouched.
    """
    arr = np.asarray(clip, dtype=np.float32).ravel()
    if arr.size == 0 or gain == 1.0:
        return arr
    return np.clip(arr * float(gain), -1.0, 1.0).astype(np.float32)


def compute_levels(clip: np.ndarray, windows: int = _LEVELS_WINDOWS) -> list[float]:
    """``windows`` evenly-spaced per-window PEAK magnitudes (0..1) across ``clip``.

    The GUI plots this as a level-over-time bar graph so a clip that is silent at
    the start / clipped in the middle is visible at a glance. Pure: splits the clip
    into ``windows`` contiguous slices and reports each slice's peak |amplitude|. An
    empty clip yields all-zero levels; a clip shorter than ``windows`` still returns
    exactly ``windows`` values (some windows empty -> 0.0).
    """
    arr = np.asarray(clip, dtype=np.float32).ravel()
    n = max(1, int(windows))
    if arr.size == 0:
        return [0.0] * n
    edges = np.linspace(0, arr.size, n + 1).astype(int)
    out: list[float] = []
    for i in range(n):
        lo, hi = int(edges[i]), int(edges[i + 1])
        seg = arr[lo:hi]
        out.append(round(float(np.max(np.abs(seg))), 4) if seg.size else 0.0)
    return out


def recordings_dir() -> str:
    """Repo-local ``debug/recordings/`` directory for saved mic/wake clips.

    Resolved from this package (``<repo>/debug/recordings``) so a clip survives the
    run for the GUI's ``GET /recordings/<file>.wav`` route and the ``play_recording``
    action. The whole ``debug/`` tree is git-ignored.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "debug" / "recordings")


def _clip_hash8(clip: np.ndarray) -> str:
    """First 8 hex of ``sha256`` over the clip's int16-LE PCM bytes (content id)."""
    import hashlib

    pcm = np.clip(np.asarray(clip, dtype=np.float32).ravel(), -1.0, 1.0)
    int16 = (pcm * 32767.0).astype("<i2").tobytes()
    return hashlib.sha256(int16).hexdigest()[:8]


def _sanitize_word(word: str) -> str:
    """A filesystem-safe per-word folder segment (no path separators / traversal).

    Keeps the word recognizable while guaranteeing it can never escape the
    recordings dir: anything that is not ``[A-Za-z0-9._-]`` collapses to ``_`` and a
    leading dot is stripped, so ``..``/``/``/`` `` can't form a traversal segment.
    Empty input -> ``"_"`` so a path segment always exists.
    """
    import re

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(word or "").strip()).lstrip(".")
    return safe or "_"


def save_recording(
    clip: np.ndarray,
    sample_rate: int,
    *,
    kind: str,
    source: str,
    word: str | None = None,
) -> tuple[str, str, str]:
    """Save a mic/wake clip as a 16 kHz mono WAV under ``debug/recordings/``.

    The shared contract for EVERY server-captured OR browser-posted diagnostic clip:
    the clip is resampled to 16 kHz mono and written as
    ``<YYYYmmdd-HHMMSS>-<source>-<hash8>.wav`` where ``source`` is ``server`` /
    ``browser`` and ``hash8`` is the first 8 hex of ``sha256`` over the int16 PCM (so
    the GUI can address it and the server can replay it).

    **Wake clips are kept as training data** in a PER-WORD subfolder
    ``debug/recordings/wake/<word>/<file>`` (ALL of them — every test is a labelled
    sample); mic-check clips stay flat in ``debug/recordings/``. Returns
    ``(path, hash8, wav_url)`` where ``wav_url`` is the ``/recordings/<rel>`` link
    (``/recordings/wake/<word>/<file>`` for a wake clip). Defensive — a disk error
    yields an empty ``path`` (the hash/url are still computed) so a diagnostic never
    crashes.
    """
    import time as _time

    from .util import wav_bytes_from_float

    clip16k = resample_to(np.asarray(clip, dtype=np.float32).ravel(), int(sample_rate), 16000)
    hash8 = _clip_hash8(clip16k)
    ts = _time.strftime("%Y%m%d-%H%M%S", _time.localtime())
    filename = "-".join([ts, source, hash8]) + ".wav"
    if kind == "wake":
        # Per-word training folder: debug/recordings/wake/<word>/<file>.
        rel = os.path.join("wake", _sanitize_word(word or "unknown"), filename)
    else:
        rel = filename
    wav_url = "/recordings/" + rel.replace(os.sep, "/")
    target = os.path.join(recordings_dir(), rel)
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as fh:
            fh.write(wav_bytes_from_float(clip16k, 16000))
        path = target
    except OSError as exc:  # disk/permission issue must not kill a diagnostic
        log.warning("recording save failed (%s): %s", target, exc)
        path = ""
    return path, hash8, wav_url


def resolve_recording(name: str) -> str | None:
    """Resolve a saved recording's BASENAME to its absolute path (traversal-safe).

    Wake clips now live in per-word subfolders (``wake/<word>/<file>``) while
    mic-check clips stay flat, so the GUI ``/recordings/`` route and the
    ``play_recording`` action address a clip by basename alone. This searches
    ``recordings_dir()`` AND its subfolders for ``<basename>.wav``, returning the
    first match (or ``None``). Only the basename is honoured — any ``..`` / path
    separators in ``name`` are discarded BEFORE the search and every candidate is
    re-checked to live under ``recordings_dir()`` (``os.path.commonpath``), so a
    crafted ``/recordings/../../etc/passwd`` can never escape the directory.
    """
    base = os.path.basename(str(name or ""))
    if not base.endswith(".wav") or base != name:
        return None
    root = os.path.realpath(recordings_dir())
    # Flat (mic-check) clip first, then any per-word wake subfolder.
    flat = os.path.join(root, base)
    if os.path.isfile(flat):
        return flat
    for dirpath, _dirs, files in os.walk(root):
        if base in files:
            cand = os.path.realpath(os.path.join(dirpath, base))
            try:
                if os.path.commonpath([root, cand]) == root:
                    return cand
            except ValueError:  # different drive on Windows -> not under root
                continue
    return None


def find_recordings(pattern: str) -> list[str]:
    """All saved recordings matching a glob ``pattern`` (flat + per-word subfolders).

    Wake clips live in ``wake/<word>/`` subfolders, so a flat
    ``glob(recordings_dir()/<pattern>)`` no longer finds them. This globs the
    recordings dir AND recurses into subfolders (``**/<pattern>``), deduplicated,
    so callers addressing a clip by hash (``*-<hash8>.wav``) or by word still match.
    """
    import glob

    root = recordings_dir()
    hits = set(glob.glob(os.path.join(root, pattern)))
    hits |= set(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    return sorted(hits)


def read_wav_float(path: str, *, target_rate: int | None = 16000) -> tuple[np.ndarray, int]:
    """Read a (mono or multi-channel) WAV into float32 PCM in [-1, 1] at ``target_rate``.

    The single reusable WAV loader shared by every clip-scoring diagnostic (wake gain
    sweep, score-clip, the new histogram / FA-eval / spectrogram eval actions). Reads
    8/16/24/32-bit PCM via :mod:`wave`, normalizes to float32 mono (multi-channel is
    averaged down), and — unless ``target_rate`` is ``None`` — resamples to it (the
    wake/embedding models expect 16 kHz). Returns ``(clip, rate)`` where ``rate`` is
    the post-resample rate (== ``target_rate`` when one was requested). Raises
    ``OSError``/``wave.Error``/``ValueError`` on an unreadable or empty file so callers
    decide how to degrade — the eval workers turn that into a clear per-file skip.
    """
    import wave

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if not raw:
        raise ValueError(f"empty WAV: {path}")
    if sampwidth == 1:  # unsigned 8-bit PCM, centered at 128
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 3:  # packed 24-bit LE -> sign-extend to int32
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints >= (1 << 23), ints - (1 << 24), ints)
        data = ints.astype(np.float32) / float(1 << 23)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"unsupported WAV sample width {sampwidth} bytes: {path}")
    if n_channels > 1:  # average channels down to mono
        data = data.reshape(-1, n_channels).mean(axis=1).astype(np.float32)
    if target_rate is not None and rate != target_rate:
        data = resample_to(data, rate, target_rate)
        rate = target_rate
    return data.astype(np.float32, copy=False), int(rate)


def list_wavs(directory: str) -> list[str]:
    """Sorted absolute paths of every ``*.wav`` directly inside ``directory``.

    The negative-corpus reader for the eval actions: the user drops wake-word-free
    WAVs into :data:`Config.negative_corpus_dir` and these are the negatives the
    histogram / FA-eval scores. Non-recursive (a flat drop-in folder), case-insensitive
    on the extension, and returns ``[]`` for a missing/empty directory (callers emit a
    clear "drop WAVs into <dir>" message rather than crashing).
    """
    import glob

    if not os.path.isdir(directory):
        return []
    hits = glob.glob(os.path.join(directory, "*.wav"))
    hits += glob.glob(os.path.join(directory, "*.WAV"))
    return sorted(set(os.path.abspath(p) for p in hits))


# A preflight is considered failed when this fraction (or more) of capture frames
# were dropped/overflowed over the short test window — a persistently-overflowing
# mic queue means capture can't keep up and the pipeline would record nothing. A
# single transient overflow on stream warm-up is tolerated below this floor.
_OVERFLOW_DROP_RATIO = 0.25


@dataclass(slots=True)
class PreflightResult:
    """Outcome of the startup audio preflight (the HARD STOP gate).

    Run BEFORE opening the GUI / starting any mic loop so a broken-audio host
    refuses to run with a clear error instead of presenting a control room that
    silently records nothing. ``ok`` is True only when a usable 16 kHz-resolvable
    capture path exists AND frames are consumed without persistent overflow.
    ``reason`` is a machine tag (``ok`` / ``no_device`` / ``rate_unresolvable`` /
    ``overflow`` / ``permission_denied`` / ``error``); ``message`` is the clear,
    actionable line shown to the user. ``device_rate`` is the rate the input device
    actually delivered, ``drop_ratio`` the measured overflow/drop fraction, and
    ``permission`` the OS mic-permission tag.
    """

    ok: bool
    reason: str
    message: str
    device_rate: int = 0
    drop_ratio: float = 0.0
    permission: str = "unknown"


def _preflight_message(reason: str, *, device_rate: int, sample_rate: int, drop_pct: int) -> str:
    """The clear, actionable hard-stop line for a failed preflight ``reason`` (pure)."""
    bypass = " (or --skip-audio-preflight to bypass)"
    if reason == "no_device":
        return (
            "Audio preflight failed (no_device): no usable microphone/input device was found "
            "— wake word & speech-to-text need a 16 kHz-capable mic. Connect an input device "
            f"and grant mic access, then retry{bypass}."
        )
    if reason == "permission_denied":
        return (
            "Audio preflight failed (permission_denied): microphone permission is DENIED for "
            "this app — enable your terminal/app in System Settings › Privacy & Security › "
            f"Microphone, then quit & reopen it and retry{bypass}."
        )
    if reason == "rate_unresolvable":
        return (
            f"Audio preflight failed (rate_unresolvable): input device runs at {device_rate} Hz "
            f"and couldn't be resampled to {sample_rate} Hz — wake word & speech-to-text need "
            f"{sample_rate // 1000} kHz. Fix the input device or rate, then retry{bypass}."
        )
    if reason == "overflow":
        return (
            f"Audio preflight failed (overflow): the mic queue is overflowing ({drop_pct}% of "
            "frames dropped) — capture can't keep up; the pipeline would record nothing. Close "
            f"other audio apps or raise the device latency/blocksize, then retry{bypass}."
        )
    return (
        "Audio preflight failed (error): the microphone could not be opened for the startup "
        f"check — see the message above. Fix the input device, then retry{bypass}."
    )


def audio_preflight(sample_rate: int = 16000, *, seconds: float = 0.5) -> PreflightResult:
    """Short real capture that HARD-STOPS a broken-audio host before the mic loop.

    Opens a ~``seconds`` input capture and verifies three things, in order, BEFORE
    any GUI / wake loop / push-to-talk starts: (a) a usable input device exists;
    (b) the device delivers audio at a rate that can be resampled to ``sample_rate``
    (16 kHz mono) — recorded as ``device_rate``; (c) frames are consumed without
    *persistent* overflow — it counts PortAudio ``input_overflow`` (the
    ``Input overflowed`` condition) plus any queue-full drops over the window and
    computes a ``drop_ratio``. A drop ratio at/above :data:`_OVERFLOW_DROP_RATIO`
    fails as ``overflow`` (a single warm-up glitch below it is tolerated).

    NEVER raises: a missing ``sounddevice`` / PortAudio, no device, a denied
    permission, an unresolvable rate, or any capture error all resolve to a failing
    :class:`PreflightResult` with a machine ``reason`` + an actionable ``message`` —
    so the caller can print it and exit non-zero rather than crash. Reuses
    :func:`mic_permission_status`: a conclusively ``denied`` permission wins
    immediately (no capture can succeed without it).
    """
    permission = mic_permission_status()
    if permission == "denied":
        return PreflightResult(
            ok=False,
            reason="permission_denied",
            message=_preflight_message(
                "permission_denied", device_rate=0, sample_rate=sample_rate, drop_pct=0
            ),
            permission=permission,
        )
    try:
        sd = _sd()
    except Exception as exc:  # noqa: BLE001 — no audio extra / no PortAudio
        return PreflightResult(
            ok=False,
            reason="error",
            message=f"Audio preflight failed (error): audio capture unavailable: {exc}",
            permission=permission,
        )
    if not mic_available():
        return PreflightResult(
            ok=False,
            reason="no_device",
            message=_preflight_message(
                "no_device", device_rate=0, sample_rate=sample_rate, drop_pct=0
            ),
            permission=permission,
        )

    device_rate = _supported_capture_rate(sd, sample_rate)
    if device_rate <= 0:
        # No positive rate could be established for the device, so there is no path
        # to resample its stream up/down to the 16 kHz the models require.
        return PreflightResult(
            ok=False,
            reason="rate_unresolvable",
            message=_preflight_message(
                "rate_unresolvable", device_rate=device_rate, sample_rate=sample_rate, drop_pct=0
            ),
            device_rate=device_rate,
            permission=permission,
        )

    total = 0
    dropped = 0
    frame_samples = 1280
    frames_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)

    def _callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
        nonlocal total, dropped
        total += 1
        # PortAudio signals it discarded buffered input (the "Input overflowed"
        # condition) via the status' input_overflow flag.
        if getattr(status, "input_overflow", False):
            dropped += 1
        try:
            frames_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            # The inbound queue is saturated — capture is outpacing the consumer,
            # exactly the "inbound mic queue full; dropping a frame" condition.
            dropped += 1

    try:
        with sd.InputStream(
            samplerate=device_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_samples,
            callback=_callback,
        ):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                try:
                    frames_q.get(timeout=0.05)
                except queue.Empty:
                    continue
    except Exception as exc:  # noqa: BLE001 — device opened/failed mid-capture
        return PreflightResult(
            ok=False,
            reason="error",
            message=f"Audio preflight failed (error): microphone error: {exc}",
            device_rate=device_rate,
            permission=permission,
        )

    # Verify a usable rate path actually exists: resampling the captured rate to the
    # pipeline rate must yield real samples. A device that delivered NOTHING (and did
    # not overflow/error) has no establishable 16 kHz path — treat as unresolvable.
    probe = resample_to(
        np.ones(max(1, device_rate // 100), dtype=np.float32), device_rate, sample_rate
    )
    if total == 0 or probe.size == 0:
        return PreflightResult(
            ok=False,
            reason="rate_unresolvable",
            message=_preflight_message(
                "rate_unresolvable", device_rate=device_rate, sample_rate=sample_rate, drop_pct=0
            ),
            device_rate=device_rate,
            permission=permission,
        )

    drop_ratio = dropped / total if total else 0.0
    drop_pct = int(round(drop_ratio * 100))
    if drop_ratio >= _OVERFLOW_DROP_RATIO:
        return PreflightResult(
            ok=False,
            reason="overflow",
            message=_preflight_message(
                "overflow", device_rate=device_rate, sample_rate=sample_rate, drop_pct=drop_pct
            ),
            device_rate=device_rate,
            drop_ratio=round(drop_ratio, 3),
            permission=permission,
        )

    return PreflightResult(
        ok=True,
        reason="ok",
        message=(
            f"Audio preflight OK — device {device_rate} Hz → {sample_rate} Hz, "
            f"{drop_pct}% frames dropped."
        ),
        device_rate=device_rate,
        drop_ratio=round(drop_ratio, 3),
        permission=permission,
    )


def record_until_silence(
    sample_rate: int,
    vad: Any,
    endpointer: Any,
    *,
    max_seconds: float,
    preroll: PreRollBuffer | None = None,
    frame_samples: int = 512,
    stop: threading.Event | None = None,
    on_debug: Any = None,
) -> np.ndarray:
    """VAD-driven capture with NO stdin — the server-side push-to-talk recorder.

    The terminal :func:`record_push_to_talk` blocks on ``input()`` (Enter to
    start/stop); driven from the browser's server-side PTT worker there is no
    interactive stdin, so it returns an empty clip immediately (the "push-to-talk
    records nothing" bug). This records hands-free instead: it opens the input
    device (at the requested rate, or the device-native rate + resample when 16 kHz
    isn't honoured), **reframes** each captured block to ``frame_samples`` (512 for
    Silero), drives ``vad.is_speech`` + the ``endpointer`` to end on a natural pause,
    and resamples the whole clip to ``sample_rate`` so the audio reaching STT is
    truly 16 kHz mono. Returns an empty array if no speech was detected.

    ``on_debug(stage, **fields)`` (optional) is called with capture/VAD telemetry so
    the debug instrument can surface where audio is lost; ``stop`` aborts cleanly.
    """
    sd = _sd()
    device_rate = _supported_capture_rate(sd, sample_rate)
    frames_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames_q.put(indata[:, 0].copy())

    collected: list[np.ndarray] = []  # at sample_rate (post-resample)
    pending = np.zeros(0, dtype=np.float32)  # buffered, not yet a full VAD frame
    spoke = False
    endpointer.reset()
    if on_debug is not None:
        on_debug("capture_start", sample_rate=sample_rate, device_rate=device_rate)
    start = time.monotonic()
    with sd.InputStream(
        samplerate=device_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_samples,
        callback=_callback,
    ):
        while time.monotonic() - start < max_seconds:
            if stop is not None and stop.is_set():
                break
            try:
                block = frames_q.get(timeout=0.1)
            except queue.Empty:
                continue
            block = resample_to(block, device_rate, sample_rate)
            collected.append(block)
            pending = np.concatenate([pending, block])
            ended = False
            while pending.size >= frame_samples:
                frame, pending = pending[:frame_samples], pending[frame_samples:]
                is_speech = vad.is_speech(frame)
                spoke = spoke or is_speech
                if on_debug is not None:
                    on_debug(
                        "vad_frame",
                        is_speech=is_speech,
                        prob=round(getattr(vad, "last_prob", 0.0), 3),
                    )
                if endpointer.update(is_speech):
                    ended = True
                    break
            if ended:
                break

    if not spoke:
        if on_debug is not None:
            on_debug(
                "no_speech",
                **capture_stats(
                    np.concatenate(collected) if collected else np.zeros(0), sample_rate
                ),
            )
        return np.zeros(0, dtype=np.float32)
    captured = np.concatenate(collected) if collected else np.zeros(0, dtype=np.float32)
    if preroll is not None:
        captured = np.concatenate([preroll.get(), captured])
    if on_debug is not None:
        on_debug("captured", **capture_stats(captured, sample_rate))
    return captured


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


class WakeDebugRecorder:
    """Tap the EXACT 16 kHz frames fed to the wake model and dump them as a WAV.

    Captures the first ``seconds`` of the post-resample, post-reframe 16 kHz mono
    frames that :func:`listen_for_wake` actually hands to ``wake.detect`` — NOT a
    separate capture — together with the per-frame wake score. On completion it
    writes a mono 16 kHz WAV to ``path`` and logs the capture stats (sample rate /
    #samples / duration / RMS / peak as a 0–100 % level) plus the MAX and MEAN wake
    score over the window, so a never-firing wake word is instantly classifiable:
    near-silent / wrong-rate / clipped audio (a capture problem) vs good audio with
    a low score (model recall). Each ``feed`` returns whether the window is still
    filling; once full the recorder writes once and goes inert. Cheap and tap-only,
    so the live wake path is unaffected when it's off.
    """

    def __init__(self, path: str, sample_rate: int, seconds: float, on_debug: Any = None) -> None:
        self.path = os.path.expanduser(path)
        self.sample_rate = sample_rate
        self.max_samples = max(1, int(sample_rate * seconds))
        self._on_debug = on_debug
        self._frames: list[np.ndarray] = []
        self._scores: list[float] = []
        self._n = 0
        self.done = False

    def feed(self, frame: np.ndarray, score: float) -> None:
        """Record one 16 kHz frame + its wake score; flush + log once the window fills."""
        if self.done:
            return
        self._frames.append(np.asarray(frame, dtype=np.float32).ravel())
        self._scores.append(float(score))
        self._n += int(np.asarray(frame).size)
        if self._n >= self.max_samples:
            self._flush()

    def _flush(self) -> None:
        self.done = True
        samples = np.concatenate(self._frames) if self._frames else np.zeros(0, dtype=np.float32)
        stats = capture_stats(samples, self.sample_rate)
        peak = float(stats["peak"])
        rms = float(stats["rms"])
        max_score = round(max(self._scores), 3) if self._scores else 0.0
        mean_score = round(float(np.mean(self._scores)), 3) if self._scores else 0.0
        wrote = self._write_wav(samples)
        fields = {
            "path": wrote,
            "sample_rate": stats["sample_rate"],
            "samples": stats["samples"],
            "duration_s": stats["duration_s"],
            "rms": rms,
            "peak": peak,
            "level_pct": int(round(min(1.0, peak) * 100)),
            "max_score": max_score,
            "mean_score": mean_score,
        }
        # stderr line so the user can see + send the WAV path even without the GUI.
        kv = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[audio:wake_debug] {kv}", file=sys.stderr)
        print(f"[audio:wake_debug] WAV saved -> {wrote}", file=sys.stderr)
        if self._on_debug is not None:
            with contextlib.suppress(Exception):
                self._on_debug("wake_debug", **fields)

    def _write_wav(self, samples: np.ndarray) -> str:
        """Write the captured frames as a 16 kHz mono 16-bit WAV; return the path."""

        from .util import wav_bytes_from_float

        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "wb") as fh:
                fh.write(wav_bytes_from_float(samples, self.sample_rate))
            return self.path
        except OSError as exc:  # disk/permission issue must not kill the wake loop
            log.warning("wake-debug WAV write failed (%s): %s", self.path, exc)
            return f"(write failed: {exc})"


def listen_for_wake(
    wake: Any,
    sample_rate: int,
    *,
    frame_samples: int = 1280,
    poll_seconds: float = 0.1,
    gain: float = 1.0,
    stop: threading.Event | None = None,
    on_debug: Any = None,
    recorder: WakeDebugRecorder | None = None,
) -> bool:
    """Block until ``wake.detect(frame)`` fires on an 80 ms (1280-sample) frame.

    openWakeWord expects 16 kHz mono, 1280-sample (80 ms) frames. The device may
    capture at its native rate (commonly 48 kHz) when 16 kHz isn't honoured, so we
    open at the supported rate, **resample** each block to ``sample_rate``, and
    **reframe** to exactly ``frame_samples`` before scoring — otherwise the wake
    model sees a wrong-rate / wrong-size frame and never fires (the reported
    symptom). Returns ``True`` when the wake word fired, or ``False`` if ``stop`` was
    set before it fired. ``on_debug`` (optional) is fed the per-evaluation wake
    max-score + model so the instrument can show why it isn't firing.

    ``gain`` (default 1.0 = no change) multiplies each 16 kHz frame, clip-protected
    to ±1.0, BEFORE :meth:`WakeWord.detect` — THE fix for the dead wake word: a quiet
    mic produces low mel energies and openWakeWord (which has no input normalization)
    collapses the score to ~0.001; lifting the gain restores the energy the model
    needs. Applied to the EXACT frame the model scores (the same frame the debug
    recorder taps), so the saved WAV reflects what the model actually sees.
    """
    sd = _sd()
    device_rate = _supported_capture_rate(sd, sample_rate)
    frames_q: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
        frames_q.put(indata[:, 0].copy())

    wake.reset()
    pending = np.zeros(0, dtype=np.float32)
    with sd.InputStream(
        samplerate=device_rate,
        channels=1,
        dtype="float32",
        blocksize=frame_samples,
        callback=_callback,
    ):
        while True:
            if stop is not None and stop.is_set():
                return False
            try:
                block = frames_q.get(timeout=poll_seconds)
            except queue.Empty:
                continue
            pending = np.concatenate([pending, resample_to(block, device_rate, sample_rate)])
            while pending.size >= frame_samples:
                frame, pending = pending[:frame_samples], pending[frame_samples:]
                # Lift the quiet capture to a level the wake model can actually score
                # (clip-protected). gain==1.0 is an identity pass (no behaviour change).
                scored_frame = apply_gain(frame, gain) if gain != 1.0 else frame
                fired = wake.detect(scored_frame)
                score = getattr(wake, "last_score", 0.0)
                # Tap the EXACT 16 kHz frame fed to the model (post-resample,
                # post-reframe, post-GAIN) into the debug recorder — this is the audio
                # the wake model truly sees, so the saved WAV is what to inspect.
                if recorder is not None and not recorder.done:
                    recorder.feed(scored_frame, score)
                if on_debug is not None:
                    on_debug(
                        "wake_score",
                        score=round(score, 3),
                        threshold=getattr(wake, "threshold", 0.5),
                        model=getattr(wake, "model_name", ""),
                    )
                if fired:
                    return True
