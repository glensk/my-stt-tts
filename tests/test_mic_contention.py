"""Server mic-check vs. the always-listen wake loop: CoreAudio device-contention fix.

Under ``--browser --wake`` the wake loop holds the input device continuously. A
server mic diagnostic (mic_check / record-replay) must take exclusive use of it: the
wake loop's ``sounddevice`` InputStream has to be fully stopped + closed AND the
thread joined BEFORE the diagnostic opens its own stream, with a short settle gap so
macOS (which frees an AUHAL input device asynchronously) hands it over — otherwise
they collide (``PaMacCore err=-50`` / paramErr) and the check records near-silence.

These tests mock ``sounddevice`` / the wake loop entirely (no real mic), and cover:

* ``_with_paused_wake`` closes + joins the wake stream and settles BEFORE running the
  paused capture (asserted by recorded open/close ordering on a mock stream);
* ``record_fixed`` retries a CoreAudio contention error (open raises -50 → retry
  succeeds; all retries fail → contention error surfaced, not a crash; an opened-but-
  silent-with-status capture is treated as contention);
* the server mic-check / record-replay message distinguishes contention ("busy") from
  a genuine permission / no-device problem.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring,redefined-outer-name,import-outside-toplevel

from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from my_stt_tts import __main__ as main_mod
from my_stt_tts import audio
from my_stt_tts.config import Config, ConfigError
from my_stt_tts.events import EventBus


# --------------------------------------------------------------------------- #
# A fake sounddevice module + InputStream that records its lifecycle calls     #
# --------------------------------------------------------------------------- #
class _FakePortAudioError(Exception):
    """Stand-in for ``sounddevice.PortAudioError``."""


class _FakeStream:
    """Records start/stop/close + (optionally) feeds frames to the callback.

    ``open_raises`` raises at ``start()`` (the AUHAL -50 race at open). ``frames``
    are the float32 mono blocks the capture callback will receive; ``status`` (when
    truthy) is reported on the FIRST callback to simulate a PortAudio input error.
    """

    def __init__(
        self,
        log: list[str],
        *,
        open_raises: BaseException | None = None,
        frames: list[np.ndarray] | None = None,
        status: Any = None,
        callback: Any = None,
        **_kw: Any,
    ) -> None:
        self._log = log
        self._open_raises = open_raises
        self._frames = frames or []
        self._status = status
        self._callback = callback
        log.append("init")

    def start(self) -> None:
        self._log.append("start")
        if self._open_raises is not None:
            raise self._open_raises
        # Deliver frames synchronously so record_fixed's drain loop sees them.
        for i, block in enumerate(self._frames):
            st = self._status if (i == 0 and self._status) else None
            self._callback(block.reshape(-1, 1), len(block), None, st)

    def stop(self) -> None:
        self._log.append("stop")

    def close(self) -> None:
        self._log.append("close")

    # context-manager parity (listen_for_wake uses `with sd.InputStream(...)`)
    def __enter__(self) -> _FakeStream:
        self.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.stop()
        self.close()


def _fake_sd(
    log: list[str],
    *,
    open_raises_seq: list[BaseException | None] | None = None,
    frames: list[np.ndarray] | None = None,
    status: Any = None,
) -> MagicMock:
    """A MagicMock posing as the ``sounddevice`` module.

    ``open_raises_seq`` supplies a per-construction ``start()`` outcome (an exception
    or None) so a retry sequence (fail, then succeed) can be modelled.
    """
    raises = list(open_raises_seq or [None])
    calls = {"n": 0}

    def _make_stream(**kw: Any) -> _FakeStream:
        idx = min(calls["n"], len(raises) - 1)
        exc = raises[idx]
        calls["n"] += 1
        return _FakeStream(log, open_raises=exc, frames=frames, status=status, **kw)

    sd = MagicMock(name="sounddevice")
    sd.InputStream.side_effect = _make_stream
    sd.PortAudioError = _FakePortAudioError
    # _supported_capture_rate probes check_input_settings; let it pass (use requested).
    sd.check_input_settings.return_value = None
    return sd


# --------------------------------------------------------------------------- #
# 1. _with_paused_wake: stream closed + thread joined + settled BEFORE fn      #
# --------------------------------------------------------------------------- #
def _controller(cfg: Config) -> main_mod._WakeController:
    return main_mod._WakeController(
        cfg,
        MagicMock(name="brain"),
        MagicMock(name="tts"),
        MagicMock(name="gate"),
        MagicMock(name="stt"),
    )


def test_with_paused_wake_releases_and_settles_before_capture() -> None:
    """The wake stream is STOPPED+CLOSED and the thread joined, and the settle delay
    elapses, BEFORE the paused capture runs — proven by the recorded event order."""
    events: list[str] = []
    cfg = Config(sample_rate=16000, mic_check_settle_s=0.05)
    ctrl = _controller(cfg)

    loop_closed = threading.Event()

    def fake_loop(*_a: Any, stop: threading.Event | None = None, **_k: Any) -> int:
        assert stop is not None
        events.append("wake_open")
        stop.wait(timeout=5)
        # The real wake stream is a context manager; closing happens on loop exit.
        events.append("wake_close")
        loop_closed.set()
        return 0

    sleeps: list[float] = []

    def fake_sleep(s: float) -> None:
        events.append(f"settle:{s}")
        sleeps.append(s)

    def paused_fn() -> None:
        events.append("capture")

    with (
        patch.object(main_mod, "run_wake_loop", side_effect=fake_loop),
        patch.object(main_mod.time, "sleep", side_effect=fake_sleep),
    ):
        ctrl.start_wake()
        assert _wait(lambda: "wake_open" in events), "wake loop never started"
        ctrl._with_paused_wake(paused_fn)
        ctrl.stop_wake()

    # Ordering: the wake stream closed, the settle slept, THEN the capture ran.
    assert events.index("wake_close") < events.index("capture")
    assert any(e.startswith("settle:") for e in events)
    assert events.index(next(e for e in events if e.startswith("settle:"))) < events.index(
        "capture"
    )
    assert sleeps and sleeps[0] == pytest.approx(0.05)
    # …and the loop was restored afterwards (a second wake_open).
    assert _wait(lambda: events.count("wake_open") == 2), "wake loop not restored"
    ctrl.stop_wake()


def test_with_paused_wake_no_settle_when_loop_not_running() -> None:
    """No wake loop → no stop/join/settle, just run fn (zero-cost passthrough)."""
    cfg = Config(sample_rate=16000, mic_check_settle_s=0.5)
    ctrl = _controller(cfg)
    ran = threading.Event()
    with patch.object(main_mod.time, "sleep") as slept:
        ctrl._with_paused_wake(ran.set)
    assert ran.is_set()
    slept.assert_not_called()


def _wait(pred: Any, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.005)
    return False


# --------------------------------------------------------------------------- #
# 2. record_fixed: retry on a CoreAudio contention error                       #
# --------------------------------------------------------------------------- #
def _good_frames() -> list[np.ndarray]:
    return [np.full(1280, 0.3, dtype=np.float32) for _ in range(4)]


def test_record_fixed_retries_open_minus50_then_succeeds() -> None:
    """First open raises AUHAL -50 (paramErr); the retry succeeds and returns audio."""
    log: list[str] = []
    err = _FakePortAudioError("||PaMacCore (AUHAL)|| Error on line 2523: err='-50'")
    sd = _fake_sd(log, open_raises_seq=[err, None], frames=_good_frames())
    with (
        patch.object(audio, "_sd", return_value=sd),
        patch.object(audio.time, "sleep"),
    ):
        clip, rate = audio.record_fixed(16000, seconds=0.05, retries=3, settle_s=0.05)
    assert clip.size > 0 and float(np.max(np.abs(clip))) > audio._SILENCE_PEAK
    assert rate == 16000
    # Two InputStreams constructed (failed attempt + successful retry).
    assert log.count("init") == 2


def test_record_fixed_all_retries_fail_raises_contention() -> None:
    """Every open raises -50 → a PortAudioError propagates (caller turns into 'busy'),
    NOT a crash with an opaque message and NOT silent success."""
    log: list[str] = []
    err = _FakePortAudioError("err='-50' paramErr")
    sd = _fake_sd(log, open_raises_seq=[err, err, err])
    with (
        patch.object(audio, "_sd", return_value=sd),
        patch.object(audio.time, "sleep"),
        pytest.raises(_FakePortAudioError),
    ):
        audio.record_fixed(16000, seconds=0.05, retries=3, settle_s=0.01)
    assert log.count("init") == 3  # exhausted all attempts


def test_record_fixed_no_retry_on_non_contention_error() -> None:
    """A non-contention open error (genuine no-device) is re-raised immediately."""
    log: list[str] = []
    sd = _fake_sd(log, open_raises_seq=[ValueError("no input device")])
    with patch.object(audio, "_sd", return_value=sd), pytest.raises(ValueError):
        audio.record_fixed(16000, seconds=0.05, retries=3, settle_s=0.01)
    assert log.count("init") == 1  # NOT retried


def test_record_fixed_silent_with_status_is_contention() -> None:
    """Stream opens but delivers silence AND flags an input-error status → treated as
    contention (raised) so it retries / surfaces as busy, not a bogus silent OK."""
    log: list[str] = []
    silent = [np.zeros(1280, dtype=np.float32) for _ in range(3)]
    # Both attempts silent+status → contention raised after exhausting retries.
    sd = _fake_sd(log, frames=silent, status="input overflow")
    with (
        patch.object(audio, "_sd", return_value=sd),
        patch.object(audio.time, "sleep"),
        pytest.raises(_FakePortAudioError),
    ):
        audio.record_fixed(16000, seconds=0.05, retries=2, settle_s=0.01)


def test_record_fixed_healthy_capture_with_stray_status_is_ok() -> None:
    """A non-silent capture that happens to flag a status is returned as-is (a stray
    overflow on a working device must NOT be misread as contention)."""
    log: list[str] = []
    sd = _fake_sd(log, frames=_good_frames(), status="output underflow")
    with patch.object(audio, "_sd", return_value=sd):
        clip, _ = audio.record_fixed(16000, seconds=0.05, retries=2, settle_s=0.0)
    assert float(np.max(np.abs(clip))) > audio._SILENCE_PEAK
    assert log.count("init") == 1


def test_capture_with_retry_returns_first_success() -> None:
    calls = {"n": 0}

    def attempt() -> str:
        calls["n"] += 1
        return "ok"

    assert audio._capture_with_retry(attempt, retries=3, settle_s=0.0) == "ok"
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# 3. is_device_contention_error classification                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("||PaMacCore (AUHAL)|| Error on line 2523: err='-50'"),
        RuntimeError("paramErr"),
        RuntimeError("device unavailable"),
    ],
)
def test_is_device_contention_error_true_for_coreaudio_markers(exc: Exception) -> None:
    assert audio.is_device_contention_error(exc) is True


@pytest.mark.parametrize(
    "exc",
    [ValueError("no input device found"), RuntimeError("permission denied"), KeyError("x")],
)
def test_is_device_contention_error_false_for_other_errors(exc: Exception) -> None:
    assert audio.is_device_contention_error(exc) is False


def test_is_device_contention_error_true_for_portaudioerror_instance() -> None:
    sd = MagicMock()
    sd.PortAudioError = _FakePortAudioError
    with patch.object(audio, "_sd", return_value=sd):
        assert audio.is_device_contention_error(_FakePortAudioError("anything")) is True


# --------------------------------------------------------------------------- #
# 4. Server mic-check / record-replay message: contention vs permission        #
# --------------------------------------------------------------------------- #
def _drain(sub: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while True:
        try:
            out.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    return out


def test_mic_check_server_contention_says_busy_not_permission() -> None:
    """When record_fixed fails with a contention error, the verdict says the mic was
    BUSY (device contention) — NOT 'check the microphone permission'."""
    cfg = Config(anthropic_api_key="sk-test")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise audio._sd().PortAudioError("err='-50' paramErr")  # contention

    bus = EventBus()
    sub = bus.subscribe()
    real_sd = MagicMock()
    real_sd.PortAudioError = _FakePortAudioError
    with (
        patch.object(audio, "_sd", return_value=real_sd),
        patch.object(audio, "record_fixed", _boom),
        patch.object(main_mod, "bus", bus),
    ):
        main_mod._run_mic_check_server(cfg)
    result = next(e for e in _drain(sub) if e["type"] == "mic_check_result")
    assert "busy" in result["message"].lower() or "contention" in result["message"].lower()
    assert "permission" not in result["message"].lower()


def test_mic_check_server_non_contention_keeps_microphone_error() -> None:
    """A genuine, non-contention capture error keeps the plain 'microphone error'
    message (no false 'busy' claim)."""
    cfg = Config(anthropic_api_key="sk-test")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise ValueError("no input device")

    bus = EventBus()
    sub = bus.subscribe()
    real_sd = MagicMock()
    real_sd.PortAudioError = _FakePortAudioError
    with (
        patch.object(audio, "_sd", return_value=real_sd),
        patch.object(audio, "record_fixed", _boom),
        patch.object(main_mod, "bus", bus),
    ):
        main_mod._run_mic_check_server(cfg)
    result = next(e for e in _drain(sub) if e["type"] == "mic_check_result")
    assert "microphone error" in result["message"].lower()
    assert "busy" not in result["message"].lower()


def test_mic_check_server_silent_success_keeps_permission_message() -> None:
    """A genuinely silent (no-error) capture still points at permission / input device
    — that real diagnostic is preserved for the actual permission/no-device case."""
    cfg = Config(anthropic_api_key="sk-test", mic_gain=2.0)
    silent = np.zeros(32000, dtype=np.float32)
    bus = EventBus()
    sub = bus.subscribe()
    with (
        patch.object(audio, "record_fixed", lambda *_a, **_k: (silent, 16000)),
        patch.object(audio, "save_recording", lambda *_a, **_k: ("/x.wav", "deadbeef", "/r/x.wav")),
        patch.object(main_mod, "bus", bus),
    ):
        main_mod._run_mic_check_server(cfg)
    result = next(e for e in _drain(sub) if e["type"] == "mic_check_result")
    assert "permission" in result["message"].lower()
    assert "busy" not in result["message"].lower()


def test_mic_record_replay_contention_says_busy() -> None:
    cfg = Config(anthropic_api_key="sk-test")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise _FakePortAudioError("err='-50'")

    bus = EventBus()
    sub = bus.subscribe()
    real_sd = MagicMock()
    real_sd.PortAudioError = _FakePortAudioError
    with (
        patch.object(audio, "_sd", return_value=real_sd),
        patch.object(audio, "record_fixed", _boom),
        patch.object(audio, "mic_permission_status", lambda: "authorized"),
        patch.object(main_mod, "bus", bus),
    ):
        main_mod._run_mic_record_replay(cfg)
    result = next(e for e in _drain(sub) if e["type"] == "mic_result")
    assert result["verdict"] == "busy"
    assert "permission" not in result["message"].lower()


# --------------------------------------------------------------------------- #
# 5. Config knobs validate                                                     #
# --------------------------------------------------------------------------- #
def test_mic_check_settle_and_retries_defaults() -> None:
    cfg = Config()
    assert cfg.mic_check_settle_s == pytest.approx(0.15)
    assert cfg.mic_check_retries == 3


@pytest.mark.parametrize("bad_settle", [-0.1, 2.5])
def test_mic_check_settle_validate_rejects_out_of_range(bad_settle: float) -> None:
    with pytest.raises(ConfigError):
        Config(anthropic_api_key="sk-test", mic_check_settle_s=bad_settle).validate()


@pytest.mark.parametrize("bad_retries", [0, 6])
def test_mic_check_retries_validate_rejects_out_of_range(bad_retries: int) -> None:
    with pytest.raises(ConfigError):
        Config(anthropic_api_key="sk-test", mic_check_retries=bad_retries).validate()


def test_mic_check_knobs_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIC_CHECK_SETTLE_S", "0.3")
    monkeypatch.setenv("MIC_CHECK_RETRIES", "2")
    cfg = Config.from_env()
    assert cfg.mic_check_settle_s == pytest.approx(0.3)
    assert cfg.mic_check_retries == 2
