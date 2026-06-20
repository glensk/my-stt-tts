"""GUI-driven server-side voice controls (wake loop + push-to-talk) and the
``voice_available`` capability gating.

These cover the browser control-room fix: the Start-Wake / Push-to-Talk buttons
must actually drive the server-side pipeline when it is available, and the
``/api/settings`` payload must honestly report whether voice can run at all. No
real mic, model, or network is used — the audio/wake/STT layers are mocked.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

from my_stt_tts import __main__ as main_mod
from my_stt_tts.config import Config
from my_stt_tts.wake import WakeUnavailable
from my_stt_tts.webui import settings_dict


def _controller() -> main_mod._WakeController:
    return main_mod._WakeController(
        Config(sample_rate=16000),
        MagicMock(name="brain"),
        MagicMock(name="tts"),
        MagicMock(name="gate"),
        MagicMock(name="stt"),
    )


def _wait_until(pred, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


# --------------------------------------------------------------------------- #
# wake_start / wake_stop toggle the server-side loop                          #
# --------------------------------------------------------------------------- #
def test_wake_start_runs_loop_and_stop_sets_event() -> None:
    started = threading.Event()
    captured_stop: dict[str, threading.Event] = {}

    def fake_loop(*_args, stop: threading.Event | None = None, **_kwargs) -> int:
        assert stop is not None
        captured_stop["e"] = stop
        started.set()
        stop.wait(timeout=5)  # exit only when the controller asks us to
        return 0

    ctrl = _controller()
    with patch.object(main_mod, "run_wake_loop", side_effect=fake_loop):
        ctrl.start_wake()
        assert started.wait(timeout=2), "wake loop thread never started"
        assert ctrl._thread is not None and ctrl._thread.is_alive()
        # Double-start is guarded: no second thread, original keeps running.
        first = ctrl._thread
        ctrl.start_wake()
        assert ctrl._thread is first
        # Stopping sets the event the loop is waiting on, and the thread winds down.
        ctrl.stop_wake()
        assert captured_stop["e"].is_set()
        assert _wait_until(lambda: not first.is_alive())
        assert ctrl._thread is None


def test_wake_stop_when_not_running_is_noop() -> None:
    ctrl = _controller()
    with patch.object(main_mod, "run_wake_loop") as loop:
        ctrl.stop_wake()  # nothing started — must not raise or call the loop
    loop.assert_not_called()


# --------------------------------------------------------------------------- #
# ptt runs ONE capture+respond                                                #
# --------------------------------------------------------------------------- #
def test_ptt_runs_one_capture_and_respond() -> None:
    ctrl = _controller()
    done = threading.Event()
    cap = main_mod._Captured(text="hello there")
    with (
        patch.object(main_mod, "_capture_ptt", return_value=cap) as capture,
        patch.object(main_mod, "_respond", side_effect=lambda *a, **k: done.set()) as respond,
    ):
        ctrl.push_to_talk()
        assert done.wait(timeout=2), "ptt worker never ran respond"
    capture.assert_called_once()
    respond.assert_called_once()
    assert respond.call_args.args[4] == "hello there"  # the transcribed text
    assert _wait_until(lambda: ctrl._ptt_busy is False)


def test_ptt_blank_capture_logs_mic_hint_and_skips_respond() -> None:
    ctrl = _controller()
    with (
        patch.object(main_mod, "_capture_ptt", return_value=main_mod._Captured(text="")),
        patch.object(main_mod, "_respond") as respond,
        patch.object(main_mod.bus, "log") as buslog,
    ):
        ctrl.push_to_talk()
        assert _wait_until(lambda: ctrl._ptt_busy is False)
    respond.assert_not_called()
    assert any("microphone permission" in str(c.args[0]) for c in buslog.call_args_list)


def test_ptt_pauses_captures_and_restores_wake_loop() -> None:
    """Push-to-talk works WHILE the wake loop is listening: the loop is paused for
    the capture+respond (so PTT owns the mic) and restarted afterwards — the same
    pause/restore pattern the record-replay diagnostic uses. No more refusal."""
    starts: list[threading.Event] = []

    def fake_loop(*_args, stop: threading.Event | None = None, **_kwargs) -> int:
        assert stop is not None
        starts.append(stop)
        stop.wait(timeout=5)  # exit only when paused/stopped
        return 0

    ctrl = _controller()
    cap = main_mod._Captured(text="hello there")
    done = threading.Event()

    def fake_respond(*_a, **_k) -> None:
        done.set()

    with (
        patch.object(main_mod, "run_wake_loop", side_effect=fake_loop),
        patch.object(main_mod, "_capture_ptt", return_value=cap) as capture,
        patch.object(main_mod, "_respond", side_effect=fake_respond) as respond,
    ):
        ctrl.start_wake()
        assert _wait_until(lambda: len(starts) == 1), "wake loop never started"
        # PTT while listening: the capture runs (loop paused), respond fires…
        ctrl.push_to_talk()
        assert done.wait(timeout=3), "ptt capture+respond never ran while wake was active"
        capture.assert_called_once()
        respond.assert_called_once()
        assert respond.call_args.args[4] == "hello there"
        # …and the wake loop was RESTORED afterwards (a fresh stop event = restart).
        assert _wait_until(lambda: len(starts) == 2, timeout=3), "wake loop not restored after ptt"
        assert _wait_until(lambda: ctrl._ptt_busy is False)
        ctrl.stop_wake()


def test_ptt_works_when_wake_off() -> None:
    """With no wake loop running, push-to-talk still captures+responds directly."""
    ctrl = _controller()
    cap = main_mod._Captured(text="hi")
    done = threading.Event()
    with (
        patch.object(main_mod, "run_wake_loop") as loop,
        patch.object(main_mod, "_capture_ptt", return_value=cap) as capture,
        patch.object(main_mod, "_respond", side_effect=lambda *a, **k: done.set()),
    ):
        ctrl.push_to_talk()
        assert done.wait(timeout=2), "ptt never ran respond with wake off"
    capture.assert_called_once()
    loop.assert_not_called()  # nothing to pause/restore when wake is off


def test_ptt_reentrancy_guarded_second_press_refused() -> None:
    """A second push-to-talk while one is still capturing is refused (no two captures
    fight over the mic) — the only remaining refusal."""
    ctrl = _controller()
    release = threading.Event()
    in_capture = threading.Event()

    def slow_capture(*_a, **_k) -> main_mod._Captured:
        in_capture.set()
        release.wait(timeout=5)  # hold the first capture open
        return main_mod._Captured(text="hi")

    with (
        patch.object(main_mod, "_capture_ptt", side_effect=slow_capture) as capture,
        patch.object(main_mod, "_respond"),
        patch.object(main_mod.bus, "log") as buslog,
    ):
        ctrl.push_to_talk()
        assert in_capture.wait(timeout=2), "first ptt never started capturing"
        ctrl.push_to_talk()  # refused: one already in flight
        time.sleep(0.1)
        release.set()
        assert _wait_until(lambda: ctrl._ptt_busy is False)
    capture.assert_called_once()  # the second press did NOT start a second capture
    assert any("already capturing" in str(c.args[0]) for c in buslog.call_args_list)


# --------------------------------------------------------------------------- #
# mic_test — capture + verdict event, works whether or not the loop is running #
# --------------------------------------------------------------------------- #
def test_run_mic_test_publishes_verdict_and_logs() -> None:
    ok = main_mod.audio.MicTestResult(
        ok=True, verdict="ok", message="Microphone OK — level 70%", level=70
    )
    with (
        patch.object(main_mod.audio, "mic_test", return_value=ok) as cap,
        patch.object(main_mod.bus, "mic_result") as micres,
        patch.object(main_mod.bus, "log"),
        patch.object(main_mod.bus, "state"),
    ):
        main_mod._run_mic_test(Config(sample_rate=16000))
    cap.assert_called_once()
    micres.assert_called_once()
    kw = micres.call_args.kwargs
    assert kw["ok"] is True and kw["verdict"] == "ok" and kw["level"] == 70


def test_run_mic_test_never_raises_on_capture_error() -> None:
    with (
        patch.object(main_mod.audio, "mic_test", side_effect=RuntimeError("boom")),
        patch.object(main_mod.bus, "mic_result") as micres,
        patch.object(main_mod.bus, "log"),
        patch.object(main_mod.bus, "state"),
    ):
        main_mod._run_mic_test(Config(sample_rate=16000))  # must not raise
    assert micres.call_args.kwargs["verdict"] == "error"


def test_controller_mic_test_runs_capture_when_idle() -> None:
    ctrl = _controller()
    done = threading.Event()
    ok = main_mod.audio.MicTestResult(ok=True, verdict="ok", message="ok", level=50)
    with (
        patch.object(main_mod.audio, "mic_test", return_value=ok) as cap,
        patch.object(main_mod.bus, "mic_result", side_effect=lambda **_k: done.set()),
        patch.object(main_mod.bus, "log"),
        patch.object(main_mod.bus, "state"),
    ):
        ctrl.mic_test()
        assert done.wait(timeout=2), "mic_test verdict never published"
    cap.assert_called_once()


def test_controller_mic_test_pauses_and_restores_wake_loop() -> None:
    """While the wake loop holds the device, a mic test stops it for the capture
    then restarts it — so the test owns the mic and the user lands back listening."""
    starts = []

    def fake_loop(*_a, stop: threading.Event | None = None, **_k) -> int:
        assert stop is not None
        starts.append(stop)
        stop.wait(timeout=5)
        return 0

    ctrl = _controller()
    ok = main_mod.audio.MicTestResult(ok=True, verdict="ok", message="ok", level=50)
    captured = threading.Event()
    with (
        patch.object(main_mod, "run_wake_loop", side_effect=fake_loop),
        patch.object(main_mod.audio, "mic_test", return_value=ok, side_effect=None) as cap,
        patch.object(main_mod.bus, "mic_result", side_effect=lambda **_k: captured.set()),
        patch.object(main_mod.bus, "log"),
        patch.object(main_mod.bus, "state"),
    ):
        cap.return_value = ok
        ctrl.start_wake()
        assert _wait_until(lambda: bool(starts))
        ctrl.mic_test()
        assert captured.wait(timeout=3), "mic test never captured"
        # The loop was restarted after the test (a second stop event was created).
        assert _wait_until(lambda: len(starts) == 2, timeout=3)
        ctrl.stop_wake()
    cap.assert_called()


# --------------------------------------------------------------------------- #
# run_wake_loop exits promptly when the stop event is set                     #
# --------------------------------------------------------------------------- #
def test_run_wake_loop_exits_on_stop_event() -> None:
    cfg = Config(sample_rate=16000)
    stop = threading.Event()

    fake_wake = MagicMock()
    fake_wake.available.return_value = True

    # ``listen_for_wake`` returns False when stop is set (its real contract); make it
    # honour the stop flag so the loop's idle wait unblocks immediately.
    def fake_listen(_wake, _sr, *, stop=None, **_kw) -> bool:  # noqa: ANN001
        return not (stop is not None and stop.is_set())

    with (
        patch("my_stt_tts.wake.WakeWord.from_config", return_value=fake_wake),
        patch("my_stt_tts.vad.SileroVad", MagicMock()),
        patch("my_stt_tts.aec.make_voiceprocessing_capture", return_value=None),
        patch("my_stt_tts.denoise.make_denoiser", return_value=None),
        patch.object(main_mod.audio, "listen_for_wake", side_effect=fake_listen),
        patch.object(main_mod.audio, "record_turn", return_value=MagicMock(size=0)),
    ):
        stop.set()  # already requested before the loop starts
        rc = main_mod.run_wake_loop(
            cfg, MagicMock(), MagicMock(), MagicMock(), MagicMock(), stop=stop
        )
    assert rc == 0  # clean exit, did not block forever


def test_run_wake_loop_stops_once_on_wakeunavailable() -> None:
    """A WakeUnavailable (e.g. openwakeword API/version mismatch) must stop the loop
    after a SINGLE clear error log — never spin the same error every frame."""
    cfg = Config(sample_rate=16000)
    fake_wake = MagicMock()
    fake_wake.available.return_value = True

    def boom(*_a, **_k):  # noqa: ANN002,ANN003
        raise WakeUnavailable("openwakeword could not load the model")

    with (
        patch("my_stt_tts.wake.WakeWord.from_config", return_value=fake_wake),
        patch("my_stt_tts.vad.SileroVad", MagicMock()),
        patch("my_stt_tts.aec.make_voiceprocessing_capture", return_value=None),
        patch("my_stt_tts.denoise.make_denoiser", return_value=None),
        patch.object(main_mod.audio, "listen_for_wake", side_effect=boom),
        patch.object(main_mod.bus, "log") as buslog,
    ):
        rc = main_mod.run_wake_loop(cfg, MagicMock(), MagicMock(), MagicMock(), MagicMock())
    assert rc == 2  # stopped, did not loop forever
    errors = [c for c in buslog.call_args_list if c.args[1:2] == ("error",)]
    assert len(errors) == 1  # logged exactly once
    assert "wake word disabled" in errors[0].args[0]


# --------------------------------------------------------------------------- #
# voice_available flag in settings_dict                                       #
# --------------------------------------------------------------------------- #
def test_settings_dict_voice_available_true() -> None:
    cfg = Config(sample_rate=16000)
    s = settings_dict(cfg, voice_available=True, voice_hint="")
    assert s["voice_available"] is True
    assert s["voice_hint"] == ""


def test_settings_dict_voice_available_false_carries_hint() -> None:
    cfg = Config(sample_rate=16000)
    s = settings_dict(cfg, voice_available=False, voice_hint="Voice off — relaunch with --wake")
    assert s["voice_available"] is False
    assert "relaunch" in s["voice_hint"]


def test_settings_dict_defaults_voice_off() -> None:
    s = settings_dict(Config(sample_rate=16000))
    assert s["voice_available"] is False
    assert s["voice_hint"] == ""


def test_settings_dict_exposes_host_app() -> None:
    # The GUI labels the server mic "uses <host_app>'s microphone permission".
    from my_stt_tts import platform as plat

    with patch.object(plat, "host_app_name", return_value="iTerm"):
        s = settings_dict(Config(sample_rate=16000))
    assert s["host_app"] == "iTerm"


# --------------------------------------------------------------------------- #
# _voice_status capability resolution                                         #
# --------------------------------------------------------------------------- #
def test_voice_status_off_without_stt() -> None:
    available, hint = main_mod._voice_status(Config(sample_rate=16000), None)
    assert available is False
    assert "--wake" in hint


def test_voice_status_off_without_wake_model() -> None:
    cfg = Config(sample_rate=16000)
    fake_wake = MagicMock()
    fake_wake.available.return_value = False
    with patch("my_stt_tts.wake.WakeWord.from_config", return_value=fake_wake):
        available, hint = main_mod._voice_status(cfg, MagicMock(name="stt"))
    assert available is False
    assert "Wake model" in hint


def test_voice_status_off_without_mic() -> None:
    cfg = Config(sample_rate=16000)
    fake_wake = MagicMock()
    fake_wake.available.return_value = True
    with (
        patch("my_stt_tts.wake.WakeWord.from_config", return_value=fake_wake),
        patch.object(main_mod.audio, "mic_available", return_value=False),
    ):
        available, hint = main_mod._voice_status(cfg, MagicMock(name="stt"))
    assert available is False
    assert "microphone" in hint.lower()


def test_voice_status_on_when_all_present() -> None:
    cfg = Config(sample_rate=16000)
    fake_wake = MagicMock()
    fake_wake.available.return_value = True
    with (
        patch("my_stt_tts.wake.WakeWord.from_config", return_value=fake_wake),
        patch.object(main_mod.audio, "mic_available", return_value=True),
    ):
        available, hint = main_mod._voice_status(cfg, MagicMock(name="stt"))
    assert available is True
    assert hint == ""


# --------------------------------------------------------------------------- #
# _run_browser wires on_action → the controller (voice on) / honest log (off) #
# --------------------------------------------------------------------------- #
class _CaptureUI:
    """Stand-in WebUI that captures the on_action callback and never blocks."""

    captured: dict[str, object] = {}

    def __init__(self, *_args, **kwargs) -> None:
        _CaptureUI.captured = dict(kwargs)
        # on_action is the 3rd positional arg in _run_browser's WebUI(...) call.
        self._on_action = _args[2] if len(_args) > 2 else kwargs.get("on_action")

    def url(self) -> str:
        return "http://127.0.0.1:8765/"

    def serve_forever(self) -> None:
        return None


def _run_browser_capture_action(stt, *, wake=False):  # noqa: ANN001, ANN202
    holder: dict[str, object] = {}

    class _UI(_CaptureUI):
        def __init__(self, *a, **k) -> None:
            super().__init__(*a, **k)
            holder["on_action"] = self._on_action
            holder["voice_available"] = k.get("voice_available")
            holder["voice_hint"] = k.get("voice_hint")

    cfg = Config(anthropic_api_key="x", sample_rate=16000)
    with patch("my_stt_tts.webui.WebUI", _UI), patch("webbrowser.open"):
        main_mod._run_browser(cfg, MagicMock(), MagicMock(), MagicMock(), stt, wake=wake, port=8765)
    return holder


def test_run_browser_on_action_drives_controller_when_voice_on() -> None:
    fake_wake = MagicMock()
    fake_wake.available.return_value = True
    with (
        patch("my_stt_tts.wake.WakeWord.from_config", return_value=fake_wake),
        patch.object(main_mod.audio, "mic_available", return_value=True),
        patch.object(main_mod._WakeController, "start_wake", autospec=True) as start,
        patch.object(main_mod._WakeController, "stop_wake", autospec=True) as stop,
        patch.object(main_mod._WakeController, "push_to_talk", autospec=True) as ptt,
    ):
        holder = _run_browser_capture_action(MagicMock(name="stt"))
        assert holder["voice_available"] is True
        on_action = holder["on_action"]
        on_action("wake_start", {})
        on_action("wake_stop", {})
        on_action("ptt", {})
    start.assert_called_once()
    stop.assert_called_once()
    ptt.assert_called_once()


def test_run_browser_on_action_logs_when_voice_off() -> None:
    # No STT → voice off → the buttons must log an honest error, not crash.
    with patch.object(main_mod.bus, "log") as buslog:
        holder = _run_browser_capture_action(None)
        assert holder["voice_available"] is False
        assert "--wake" in str(holder["voice_hint"])
        on_action = holder["on_action"]
        on_action("wake_start", {})
        on_action("ptt", {})
    msgs = [str(c.args[0]) for c in buslog.call_args_list]
    assert any("unavailable" in m for m in msgs)
