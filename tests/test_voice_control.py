"""Tests for the mac-voice control daemon (fake sessions/listeners; no audio, no network)."""

from __future__ import annotations

import plistlib
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar

import pytest

from my_stt_tts import voice_control as vc

FAST = {"settle": 0.0, "cooldown": 0.0}


class FakeSession:
    """A conversation that ends when ``end`` is called (or when told to hang)."""

    def __init__(self, idle: float = 0.0, hang: bool = False) -> None:
        self.started = False
        self.ended = threading.Event()
        self.idle = idle
        self.hang = hang
        self.release = threading.Event()

    def start(self) -> None:
        self.started = True

    def end(self) -> None:
        self.ended.set()

    def wait(self) -> str | None:
        if self.hang:
            self.release.wait(30)
        self.ended.wait(30)
        return "conv_fake"

    def idle_for(self) -> float:
        return self.idle


class FakeListener:
    instances: ClassVar[list[FakeListener]] = []
    log: ClassVar[list[str]] = []

    def __init__(self, on_wake: Callable[[], None]) -> None:
        self.on_wake = on_wake
        self.running = False
        FakeListener.instances.append(self)

    def start(self) -> None:
        self.running = True
        FakeListener.log.append("armed")

    def stop(self) -> bool:
        self.running = False
        FakeListener.log.append("disarmed")
        return True


def _wait_for(pred: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


@pytest.fixture(name="made")
def _made(tmp_path: Path) -> tuple[vc.VoiceDaemon, list[FakeSession], list[str]]:
    FakeListener.instances.clear()
    FakeListener.log.clear()
    sessions: list[FakeSession] = []
    said: list[str] = []

    def factory() -> FakeSession:
        sessions.append(FakeSession())
        return sessions[-1]

    def announce(text: str) -> None:
        said.append(text)
        FakeListener.log.append(f"say {text}")

    daemon = vc.VoiceDaemon(
        factory,
        listener_factory=FakeListener,
        announce=announce,
        notify=lambda: None,
        state_dir=tmp_path,
        timing=FAST,
    )
    return daemon, sessions, said


def test_toggle_starts_then_stops_with_announcements(made) -> None:
    daemon, sessions, said = made
    daemon.handle("toggle")
    _wait_for(lambda: daemon.state == "talking")
    assert said == ["voice on"] and sessions[0].started
    daemon.handle("toggle")
    _wait_for(lambda: daemon.state == "idle")
    assert sessions[0].ended.is_set()
    assert said == ["voice on", "voice off"]


def test_wake_rearms_only_after_voice_off(made) -> None:
    daemon, _sessions, _said = made
    daemon.submit("arm")
    daemon.drain()
    FakeListener.instances[0].on_wake()  # the wake word fired
    _wait_for(lambda: daemon.state == "talking")
    assert not FakeListener.instances[0].running  # mic released before the session took it
    daemon.handle("off")
    _wait_for(lambda: daemon.state == "idle")
    daemon.drain()
    assert FakeListener.log == ["armed", "disarmed", "say voice on", "say voice off", "armed"]


def test_concurrent_toggles_are_linearised(made) -> None:
    daemon, sessions, _said = made
    threads = [threading.Thread(target=daemon.handle, args=("toggle",)) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    daemon.drain()
    _wait_for(lambda: daemon.state in ("idle", "talking"))
    daemon.drain()
    assert len(sessions) <= 2  # never two sessions alive at once
    assert sum(not s.ended.is_set() for s in sessions) <= 1


def test_idle_timeout_hangs_up(tmp_path: Path) -> None:
    session = FakeSession(idle=999.0)
    daemon = vc.VoiceDaemon(
        lambda: session,
        announce=lambda _t: None,
        notify=lambda: None,
        state_dir=tmp_path,
        timing=FAST | {"idle_timeout": 0.5},
    )
    daemon.handle("on")
    _wait_for(session.ended.is_set)
    _wait_for(lambda: daemon.state == "idle")


def test_stuck_session_triggers_clean_restart(tmp_path: Path, monkeypatch) -> None:
    fatal = threading.Event()
    monkeypatch.setattr(vc.VoiceDaemon, "_fatal", lambda self, why: fatal.set())
    session = FakeSession(hang=True)
    daemon = vc.VoiceDaemon(
        lambda: session,
        announce=lambda _t: None,
        notify=lambda: None,
        state_dir=tmp_path,
        timing=FAST | {"stop_deadline": 0.3},
    )
    daemon.handle("on")
    _wait_for(lambda: daemon.state == "talking")
    daemon.handle("off")
    _wait_for(fatal.is_set)
    session.release.set()


def test_off_right_after_on_ends_the_new_session(made) -> None:
    daemon, sessions, _said = made
    daemon.handle("on")
    daemon.handle("off")
    daemon.drain()
    _wait_for(lambda: daemon.state == "idle")
    assert sessions and sessions[0].ended.is_set()


def test_wake_preference_persists(made, tmp_path: Path) -> None:
    daemon, _s, _a = made
    daemon.handle("wake off")
    daemon.drain()
    assert daemon.status()["wake"] is False
    again = vc.VoiceDaemon(
        FakeSession, listener_factory=FakeListener, notify=lambda: None, state_dir=tmp_path
    )
    assert again.wake_enabled is False


def test_failed_start_returns_to_idle(tmp_path: Path) -> None:
    said: list[str] = []

    def broken() -> FakeSession:
        raise RuntimeError("no network")

    daemon = vc.VoiceDaemon(
        broken, announce=said.append, notify=lambda: None, state_dir=tmp_path, timing=FAST
    )
    daemon.handle("on")
    daemon.drain()
    assert daemon.state == "idle" and said == ["voice on", "voice failed"]


def test_silent_startup_failure_is_announced_with_its_reason(tmp_path: Path) -> None:
    """The SDK died before the conversation opened (quota): "voice failed: …", not "voice off"."""
    said: list[str] = []

    class FailedSession(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.start_failure = "ElevenLabs quota exceeded"

    session = FailedSession()
    daemon = vc.VoiceDaemon(
        lambda: session, announce=said.append, notify=lambda: None, state_dir=tmp_path, timing=FAST
    )
    daemon.handle("on")
    _wait_for(lambda: daemon.state == "talking")
    session.end()  # what VoiceSession does once its start watchdog saw the failure
    _wait_for(lambda: daemon.state == "idle")
    assert said == ["voice on", "voice failed: ElevenLabs quota exceeded"]


def test_failure_announcement_without_a_reason() -> None:
    assert vc.failure_announcement(None) == "voice failed"
    assert vc.failure_announcement(FakeSession()) == "voice failed"


def test_unknown_command_is_rejected(made) -> None:
    daemon, _s, _a = made
    assert daemon.handle("dance")["ok"] is False


def test_socket_roundtrip_and_status_file(made, tmp_path: Path) -> None:
    daemon, _s, _a = made
    short = Path(tempfile.mkdtemp(prefix="mv", dir="/tmp"))  # AF_UNIX paths max 104 chars
    path = short / "c.sock"
    stop = threading.Event()
    server = threading.Thread(target=vc.serve, args=(daemon, path, stop), daemon=True)
    server.start()
    _wait_for(path.exists)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    reply = vc.send("status", path)
    assert reply is not None and reply["state"] == "idle"
    with pytest.raises(RuntimeError):
        vc.serve(daemon, path, threading.Event())  # a second daemon refuses to start
    vc.send("on", path)
    _wait_for(lambda: daemon.state == "talking")
    assert '"talking"' in (tmp_path / "status.json").read_text()
    stop.set()
    server.join(2)
    assert not path.exists()
    assert vc.send("status", path) is None
    shutil.rmtree(short, ignore_errors=True)


def test_launch_agent_plist_runs_the_daemon(tmp_path: Path) -> None:
    data = plistlib.loads(vc.launch_agent_plist(Path("/x/mac-voice"), tmp_path))
    assert data["Label"] == "com.albert.mac-voice"
    assert data["ProgramArguments"] == ["/x/mac-voice", "-d"]
    assert data["WorkingDirectory"] == "/x"
    assert data["KeepAlive"] is True and data["RunAtLoad"] is True
    assert data["StandardErrorPath"] == str(tmp_path / "daemon.log")


# -- ready chime -----------------------------------------------------------------------------
def test_ready_sound_defaults_and_switches_off() -> None:
    assert vc.ready_sound({}) == vc.READY_SOUND == "/System/Library/Sounds/Glass.aiff"
    assert vc.ready_sound({"MAC_VOICE_READY_SOUND": " /tmp/ping.aiff "}) == "/tmp/ping.aiff"
    for off in ("", "0", "off", "OFF", "  "):
        assert vc.ready_sound({"MAC_VOICE_READY_SOUND": off}) is None


def test_play_ready_sound_runs_afplay_without_waiting() -> None:
    calls: list[list[str]] = []

    def popen(argv: list[str], **_kw: object) -> None:
        calls.append(argv)

    vc.play_ready_sound({}, popen=popen)
    assert calls == [["/usr/bin/afplay", vc.READY_SOUND]]
    vc.play_ready_sound({"MAC_VOICE_READY_SOUND": "off"}, popen=popen)
    assert len(calls) == 1  # switched off: nothing played


def test_play_ready_sound_ignores_failures() -> None:
    def popen(argv: list[str], **_kw: object) -> None:
        raise FileNotFoundError(argv[0])

    vc.play_ready_sound({}, popen=popen)  # never raises


class _QuietDaemon:
    """Stands in for VoiceDaemon so daemon_main touches no real state dir."""

    wake_enabled = False

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.events: list[str] = []

    def submit(self, event: str, arg: object = None) -> None:
        del arg
        self.events.append(event)

    def publish(self) -> None:
        pass

    def shutdown(self) -> None:
        pass


@pytest.mark.parametrize("fails", [False, True])
def test_daemon_main_chimes_once_it_listens(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fails: bool
) -> None:
    order: list[str] = []

    def chime() -> None:
        order.append("chime")
        if fails:
            raise RuntimeError("no audio")

    monkeypatch.setattr(vc, "_build_bridge", lambda: (None, None))
    monkeypatch.setattr(vc, "VoiceDaemon", _QuietDaemon)
    monkeypatch.setattr(vc, "STATE_DIR", tmp_path)
    monkeypatch.setattr(vc, "socket_path", lambda: tmp_path / "control.sock")
    monkeypatch.setattr(vc, "serve", lambda *_a: order.append("serve"))
    monkeypatch.setattr(vc, "refresh_menu_bar", lambda: None)
    monkeypatch.setattr(vc.signal, "signal", lambda *_a: None)
    assert vc.daemon_main(lambda **_kw: FakeSession(), wake=False, chime=chime) == 0
    assert order == ["chime", "serve"]
