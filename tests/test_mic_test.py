"""Server-side microphone test: the verdict mapping + the capture wrapper, and
the GUI ``mic_test`` action / event plumbing.

The user's "I can't be heard" bug needs a way to confirm the SERVER mic
(``sounddevice``) actually captures audio. :func:`audio.mic_test_verdict` is the
pure decision (working / silent / no-device / error); :func:`audio.mic_test`
wraps a short real capture. No real microphone is used — the capture layer is
faked so the loud / silent / error verdicts are all asserted deterministically.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from my_stt_tts import audio


# --------------------------------------------------------------------------- #
# Pure verdict mapping                                                         #
# --------------------------------------------------------------------------- #
def test_verdict_ok_when_loud() -> None:
    r = audio.mic_test_verdict(captured=True, rms=0.3, peak=0.8)
    assert r.ok is True
    assert r.verdict == "ok"
    assert r.level == 80
    assert "Microphone OK" in r.message
    assert "80%" in r.message


def test_verdict_silent_points_at_macos_permission() -> None:
    r = audio.mic_test_verdict(captured=True, rms=0.0, peak=0.0005)
    assert r.ok is False
    assert r.verdict == "silent"
    assert "No audio" in r.message
    # The most likely macOS cause must be named explicitly.
    assert "Privacy & Security" in r.message
    assert "Microphone" in r.message


def test_verdict_no_device() -> None:
    r = audio.mic_test_verdict(captured=False, rms=0.0, peak=0.0)
    assert r.ok is False
    assert r.verdict == "no_device"
    assert "No microphone" in r.message


def test_verdict_error_wins_and_is_verbatim() -> None:
    r = audio.mic_test_verdict(
        captured=True, rms=1.0, peak=1.0, error="PortAudio: device unplugged"
    )
    assert r.ok is False
    assert r.verdict == "error"
    assert r.message == "PortAudio: device unplugged"


def test_level_is_clamped_0_100() -> None:
    assert audio.mic_test_verdict(captured=True, rms=0.0, peak=5.0).level == 100
    assert audio.mic_test_verdict(captured=True, rms=0.0, peak=-1.0).verdict == "silent"


# --------------------------------------------------------------------------- #
# Capture wrapper — fake sounddevice feeds frames through the callback         #
# --------------------------------------------------------------------------- #
class _FakeStream:
    """A context-manager InputStream that fires the callback with preset frames."""

    def __init__(self, frames: list[np.ndarray], **kwargs: Any) -> None:
        self._frames = frames
        self._cb = kwargs["callback"]

    def __enter__(self) -> _FakeStream:
        for f in self._frames:
            # sounddevice hands (n, channels) float32; the helper takes column 0.
            self._cb(f.reshape(-1, 1), len(f), None, None)
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _fake_sd(frames: list[np.ndarray]) -> MagicMock:
    sd = MagicMock(name="sounddevice")
    sd.InputStream.side_effect = lambda **kw: _FakeStream(frames, **kw)
    return sd


def test_mic_test_reports_ok_on_loud_capture() -> None:
    loud = [np.full(1280, 0.6, dtype=np.float32) for _ in range(3)]
    with (
        patch.object(audio, "_sd", return_value=_fake_sd(loud)),
        patch.object(audio, "mic_available", return_value=True),
    ):
        r = audio.mic_test(16000, seconds=0.0)  # deadline immediately past → no busy loop
    assert r.ok is True
    assert r.verdict == "ok"
    assert r.level == 60


def test_mic_test_reports_silent_on_zero_capture() -> None:
    silent = [np.zeros(1280, dtype=np.float32) for _ in range(3)]
    # Pin the permission so the verdict message is deterministic across OSes: on an
    # authorized Mac a silent capture reads "granted but no audio" (a device issue);
    # with no conclusive permission it falls to the generic Privacy & Security hint.
    with (
        patch.object(audio, "_sd", return_value=_fake_sd(silent)),
        patch.object(audio, "mic_available", return_value=True),
        patch.object(audio, "mic_permission_status", return_value="unavailable"),
    ):
        r = audio.mic_test(16000, seconds=0.0)
    assert r.ok is False
    assert r.verdict == "silent"
    assert "Privacy & Security" in r.message


def test_mic_test_reports_error_when_stream_raises() -> None:
    sd = MagicMock(name="sounddevice")
    sd.InputStream.side_effect = OSError("PortAudio: -9996 invalid device")
    with (
        patch.object(audio, "_sd", return_value=sd),
        patch.object(audio, "mic_available", return_value=True),
    ):
        r = audio.mic_test(16000, seconds=0.0)
    assert r.ok is False
    assert r.verdict == "error"
    assert "PortAudio" in r.message


def test_mic_test_no_device_when_mic_unavailable() -> None:
    with (
        patch.object(audio, "_sd", return_value=MagicMock()),
        patch.object(audio, "mic_available", return_value=False),
    ):
        r = audio.mic_test(16000, seconds=0.0)
    assert r.verdict == "no_device"


def test_mic_test_unavailable_when_sounddevice_missing() -> None:
    with patch.object(audio, "_sd", side_effect=ImportError("no module named sounddevice")):
        r = audio.mic_test(16000, seconds=0.0)
    assert r.ok is False
    assert r.verdict == "error"
    assert "unavailable" in r.message


def test_mic_test_never_raises_on_unexpected_failure() -> None:
    # Even a surprise inside mic_available is contained (mic_test never crashes).
    with patch.object(audio, "_sd", side_effect=RuntimeError("boom")):
        r = audio.mic_test(16000, seconds=0.0)
    assert r.verdict == "error"
