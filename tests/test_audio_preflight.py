"""Startup audio preflight HARD STOP — the broken-audio gate.

Two layers are covered, both WITHOUT a real microphone (``sounddevice`` /
capture is faked entirely):

1. :func:`audio.audio_preflight` — the pure-ish detector: OK on a fake 16 kHz
   device; a hard-stop ``PreflightResult`` on an unresolvable rate, a persistently
   overflowing mic queue (high drop ratio), no device, denied permission, and a
   capture error. It NEVER raises.
2. :func:`__main__.main` wiring — the mic-using modes (``--wake``, ``--browser
   --wake``, ``--browser --browser-audio``, default push-to-talk) are GATED: a
   failing preflight makes ``main()`` return non-zero and does NOT open the GUI /
   start capture. Mic-less modes (``--type``, ``--text``, plain ``--browser``) skip
   it, and ``--skip-audio-preflight`` bypasses the gate.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from my_stt_tts import __main__ as main_mod
from my_stt_tts import audio
from my_stt_tts.config import Config


# --------------------------------------------------------------------------- #
# Fake sounddevice: an InputStream that fires the callback with preset frames  #
# and an optional per-frame status (carrying input_overflow).                  #
# --------------------------------------------------------------------------- #
class _FakeStatus:
    """Stand-in for sounddevice.CallbackFlags exposing ``input_overflow``."""

    def __init__(self, *, input_overflow: bool = False) -> None:
        self.input_overflow = input_overflow


class _FakeStream:
    """Context-manager InputStream firing the callback with (frame, status) pairs."""

    def __init__(
        self, frames: list[np.ndarray], statuses: list[_FakeStatus], **kwargs: Any
    ) -> None:
        self._frames = frames
        self._statuses = statuses
        self._cb = kwargs["callback"]

    def __enter__(self) -> _FakeStream:
        for frame, status in zip(self._frames, self._statuses, strict=False):
            self._cb(frame.reshape(-1, 1), len(frame), None, status)
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def _fake_sd(
    frames: list[np.ndarray],
    *,
    overflow_frames: int = 0,
    supported_rate: int = 16000,
) -> MagicMock:
    statuses = [_FakeStatus(input_overflow=i < overflow_frames) for i in range(len(frames))]
    sd = MagicMock(name="sounddevice")
    sd.InputStream.side_effect = lambda **kw: _FakeStream(frames, statuses, **kw)
    # _supported_capture_rate probes check_input_settings; make it honour the rate.
    if supported_rate == 16000:
        sd.check_input_settings.return_value = None
    else:
        sd.check_input_settings.side_effect = RuntimeError("rate not honoured")
        sd.query_devices.return_value = {"default_samplerate": float(supported_rate)}
    return sd


# --------------------------------------------------------------------------- #
# audio_preflight — the detector                                               #
# --------------------------------------------------------------------------- #
def test_preflight_ok_on_fake_16k_device() -> None:
    # The synchronous fake fires every callback before the consumer drains, so keep
    # the burst within the bounded inbound queue (maxsize 8) — a clean 16 kHz device.
    frames = [np.full(1280, 0.4, dtype=np.float32) for _ in range(6)]
    with (
        patch.object(audio, "_sd", return_value=_fake_sd(frames)),
        patch.object(audio, "mic_available", return_value=True),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
    ):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is True
    assert r.reason == "ok"
    assert r.device_rate == 16000
    assert r.drop_ratio == 0.0
    assert r.permission == "authorized"


def test_preflight_overflow_when_queue_floods() -> None:
    # Every frame carries input_overflow → drop ratio = 1.0 → hard stop.
    frames = [np.full(1280, 0.4, dtype=np.float32) for _ in range(10)]
    with (
        patch.object(audio, "_sd", return_value=_fake_sd(frames, overflow_frames=10)),
        patch.object(audio, "mic_available", return_value=True),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
    ):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is False
    assert r.reason == "overflow"
    assert r.drop_ratio >= audio._OVERFLOW_DROP_RATIO
    assert "overflow" in r.message.lower()
    assert "%" in r.message  # the drop percentage is shown
    assert "--skip-audio-preflight" in r.message


def test_preflight_tolerates_a_single_warmup_overflow() -> None:
    # 1 transient overflow out of 8 frames (12.5%) is below the floor → still OK
    # (8 frames fit the bounded queue, so no spurious queue-full drops are counted).
    frames = [np.full(1280, 0.4, dtype=np.float32) for _ in range(8)]
    with (
        patch.object(audio, "_sd", return_value=_fake_sd(frames, overflow_frames=1)),
        patch.object(audio, "mic_available", return_value=True),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
    ):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is True
    assert r.reason == "ok"
    assert 0.0 < r.drop_ratio < audio._OVERFLOW_DROP_RATIO


def test_preflight_48k_device_resamples_and_passes() -> None:
    # A 48 kHz device is fine: resample_to maps it to 16 kHz. NOT unresolvable.
    frames = [np.full(3840, 0.4, dtype=np.float32) for _ in range(10)]
    with (
        patch.object(audio, "_sd", return_value=_fake_sd(frames, supported_rate=48000)),
        patch.object(audio, "mic_available", return_value=True),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
    ):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is True
    assert r.device_rate == 48000


def test_preflight_rate_unresolvable_when_no_positive_rate() -> None:
    # _supported_capture_rate yields a non-positive rate → no path to 16 kHz.
    frames = [np.zeros(1280, dtype=np.float32) for _ in range(3)]
    with (
        patch.object(audio, "_sd", return_value=_fake_sd(frames)),
        patch.object(audio, "mic_available", return_value=True),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
        patch.object(audio, "_supported_capture_rate", return_value=0),
    ):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is False
    assert r.reason == "rate_unresolvable"
    assert "16000 Hz" in r.message or "16 kHz" in r.message
    assert "--skip-audio-preflight" in r.message


def test_preflight_rate_unresolvable_when_no_frames_delivered() -> None:
    # Device opened but delivered NO frames (and no overflow/error): no usable path.
    with (
        patch.object(audio, "_sd", return_value=_fake_sd([], supported_rate=48000)),
        patch.object(audio, "mic_available", return_value=True),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
    ):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is False
    assert r.reason == "rate_unresolvable"


def test_preflight_no_device() -> None:
    with (
        patch.object(audio, "_sd", return_value=MagicMock()),
        patch.object(audio, "mic_available", return_value=False),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
    ):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is False
    assert r.reason == "no_device"
    assert "--skip-audio-preflight" in r.message


def test_preflight_permission_denied_wins_immediately() -> None:
    # A conclusively denied permission short-circuits before any capture attempt.
    with patch.object(audio, "mic_permission_status", return_value="denied"):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is False
    assert r.reason == "permission_denied"
    assert "DENIED" in r.message
    assert r.permission == "denied"


def test_preflight_error_when_stream_raises_never_propagates() -> None:
    sd = MagicMock(name="sounddevice")
    sd.check_input_settings.return_value = None
    sd.InputStream.side_effect = OSError("PortAudio: -9996 invalid device")
    with (
        patch.object(audio, "_sd", return_value=sd),
        patch.object(audio, "mic_available", return_value=True),
        patch.object(audio, "mic_permission_status", return_value="authorized"),
    ):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is False
    assert r.reason == "error"
    assert "PortAudio" in r.message


def test_preflight_error_when_sounddevice_missing() -> None:
    with (
        patch.object(audio, "_sd", side_effect=ImportError("no sounddevice")),
        patch.object(audio, "mic_permission_status", return_value="unavailable"),
    ):
        r = audio.audio_preflight(16000, seconds=0.0)
    assert r.ok is False
    assert r.reason == "error"
    assert "unavailable" in r.message


# --------------------------------------------------------------------------- #
# main() wiring — the HARD STOP gate                                           #
# --------------------------------------------------------------------------- #
class _ExplodingUI:
    """A WebUI that fails the test if it is ever constructed (must not open on fail)."""

    def __init__(self, *_a: Any, **_k: Any) -> None:
        raise AssertionError("WebUI must NOT be opened when the preflight fails")


def _fail_result() -> audio.PreflightResult:
    return audio.PreflightResult(
        ok=False,
        reason="overflow",
        message="Audio preflight failed (overflow): ... --skip-audio-preflight to bypass.",
        device_rate=48000,
        drop_ratio=0.9,
    )


def _ok_result() -> audio.PreflightResult:
    return audio.PreflightResult(ok=True, reason="ok", message="OK", device_rate=16000)


def _patched_main(
    argv: list[str], *, preflight: audio.PreflightResult
) -> tuple[int, MagicMock, MagicMock, MagicMock]:
    """Run main() with the heavy machinery stubbed and a fixed preflight result.

    Returns ``(exit_code, preflight_mock, wake_loop_mock, terminal_mock)`` so each
    test can assert the exit code AND whether capture / the GUI was ever started.
    """
    with (
        patch.object(main_mod.Config, "from_env", return_value=Config(anthropic_api_key="x")),
        patch.object(main_mod.audio, "audio_preflight", return_value=preflight) as pf,
        patch("my_stt_tts.webui.WebUI", _ExplodingUI),
        patch("webbrowser.open"),
        patch.object(main_mod, "Brain", MagicMock()),
        patch.object(main_mod, "TTSRouter", MagicMock()),
        patch.object(main_mod, "run_wake_loop", return_value=0) as wake_loop,
        patch.object(main_mod, "_run_terminal_modes") as terminal,
        # Block the heavy STT import so a passing-preflight path stays light.
        patch("my_stt_tts.stt.make_transcriber", return_value=MagicMock(), create=True),
    ):
        rc = main_mod.main(argv)
    return rc, pf, wake_loop, terminal


def test_main_wake_hard_stops_without_opening_gui() -> None:
    rc, pf, wake_loop, _terminal = _patched_main(["--wake"], preflight=_fail_result())
    assert rc == 3  # non-zero hard-stop exit
    pf.assert_called_once()
    wake_loop.assert_not_called()  # capture never started


def test_main_browser_wake_hard_stops() -> None:
    rc, pf, _wl, _terminal = _patched_main(["--browser", "--wake"], preflight=_fail_result())
    assert rc == 3
    pf.assert_called_once()  # _ExplodingUI would have raised if the GUI opened


def test_main_browser_audio_hard_stops() -> None:
    rc, pf, _wl, _terminal = _patched_main(
        ["--browser", "--browser-audio"], preflight=_fail_result()
    )
    assert rc == 3
    pf.assert_called_once()


def test_main_default_ptt_hard_stops() -> None:
    rc, pf, _wl, terminal = _patched_main([], preflight=_fail_result())
    assert rc == 3
    pf.assert_called_once()
    terminal.assert_not_called()  # the push-to-talk loop never started


def test_main_wake_proceeds_when_preflight_ok() -> None:
    rc, pf, wake_loop, _terminal = _patched_main(["--wake"], preflight=_ok_result())
    assert rc == 0
    pf.assert_called_once()
    wake_loop.assert_called_once()  # capture proceeded after a passing preflight


def test_main_type_mode_skips_preflight() -> None:
    rc, pf, _wl, _terminal = _patched_main(["--type"], preflight=_fail_result())
    # Even with a FAILING preflight, --type must run (mic-less) and never call it.
    assert rc == 0
    pf.assert_not_called()


def test_main_text_mode_skips_preflight() -> None:
    rc, pf, _wl, _terminal = _patched_main(["--text", "hello"], preflight=_fail_result())
    assert rc == 0
    pf.assert_not_called()


def test_main_plain_browser_skips_preflight() -> None:
    # --browser without --wake / --browser-audio is state/transcript only (no mic).
    with (
        patch.object(main_mod.Config, "from_env", return_value=Config(anthropic_api_key="x")),
        patch.object(main_mod.audio, "audio_preflight", return_value=_fail_result()) as pf,
        patch.object(main_mod, "_run_browser", return_value=0) as run_browser,
        patch.object(main_mod, "Brain", MagicMock()),
        patch.object(main_mod, "TTSRouter", MagicMock()),
    ):
        rc = main_mod.main(["--browser"])
    assert rc == 0
    pf.assert_not_called()
    run_browser.assert_called_once()


def test_main_skip_flag_bypasses_a_failing_preflight() -> None:
    # --skip-audio-preflight must let a mic mode run even if the preflight WOULD fail.
    with (
        patch.object(main_mod.Config, "from_env", return_value=Config(anthropic_api_key="x")),
        patch.object(main_mod.audio, "audio_preflight", return_value=_fail_result()) as pf,
        patch.object(main_mod, "Brain", MagicMock()),
        patch.object(main_mod, "TTSRouter", MagicMock()),
        patch.object(main_mod, "run_wake_loop", return_value=0) as wake_loop,
        patch("my_stt_tts.stt.make_transcriber", return_value=MagicMock(), create=True),
    ):
        rc = main_mod.main(["--wake", "--skip-audio-preflight"])
    assert rc == 0
    pf.assert_not_called()  # gate short-circuits before calling the preflight
    wake_loop.assert_called_once()
