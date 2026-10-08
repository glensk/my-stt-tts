"""Tests for the mac-voice control daemon (fake sessions/listeners; no audio, no network)."""

from __future__ import annotations

import plistlib
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from my_stt_tts import voice_control as vc


class FakeSession:
    """A conversation that ends when ``end`` is called (or never, until then)."""

    def __init__(self, idle: float = 0.0) -> None:
        self.started = False
        self.ended = threading.Event()
        self.idle = idle

    def start(self) -> None:
        self.started = True

    def end(self) -> None:
        self.ended.set()

    def wait(self) -> str | None:
        self.ended.wait(5)
        return "conv_fake"

    def idle_for(self) -> float:
        return self.idle


class FakeListener:
    instances: list[FakeListener] = []  # noqa: RUF012 — test registry

    def __init__(self, on_wake: Callable[[], None]) -> None:
        self.on_wake = on_wake
        self.running = False
        FakeListener.instances.append(self)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


def _wait_for(pred: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        assert time.monotonic() < deadline, "condition not reached"
        time.sleep(0.01)


@pytest.fixture(name="made")
def _made(tmp_path: Path) -> tuple[vc.VoiceDaemon, list[FakeSession], list[str]]:
    FakeListener.instances.clear()
    sessions: list[FakeSession] = []
    said: list[str] = []

    def factory() -> FakeSession:
        sessions.append(FakeSession())
        return sessions[-1]

    daemon = vc.VoiceDaemon(
        factory,
        listener_factory=FakeListener,
        announce=said.append,
        notify=lambda: None,
        state_dir=tmp_path,
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
    _wait_for(lambda: said == ["voice on", "voice off"])


def test_wake_listener_released_before_session_and_rearmed_after(made) -> None:
    daemon, _sessions, _said = made
    daemon.arm_wake()
    first = FakeListener.instances[0]
    assert first.running
    first.on_wake()  # the wake word fired
    _wait_for(lambda: daemon.state == "talking")
    assert not first.running  # mic released before the session took it
    daemon.handle("off")
    _wait_for(lambda: len(FakeListener.instances) == 2 and FakeListener.instances[1].running)


def test_idle_timeout_hangs_up(tmp_path: Path) -> None:
    session = FakeSession(idle=999.0)
    daemon = vc.VoiceDaemon(
        lambda: session, announce=lambda _t: None, notify=lambda: None, state_dir=tmp_path
    )
    daemon.idle_timeout = 1.0
    daemon.handle("on")
    _wait_for(session.ended.is_set)
    _wait_for(lambda: daemon.state == "idle")


def test_off_during_starting_cancels_the_session(tmp_path: Path) -> None:
    gate = threading.Event()
    session = FakeSession()

    def slow_factory() -> FakeSession:
        gate.wait(2)
        return session

    daemon = vc.VoiceDaemon(
        slow_factory, announce=lambda _t: None, notify=lambda: None, state_dir=tmp_path
    )
    daemon.handle("on")
    _wait_for(lambda: daemon.state == "starting")
    daemon.handle("off")
    gate.set()
    _wait_for(session.ended.is_set)


def test_wake_preference_persists(made, tmp_path: Path) -> None:
    daemon, _s, _a = made
    reply = daemon.handle("wake off")
    assert reply["ok"] and reply["wake"] is False
    again = vc.VoiceDaemon(
        FakeSession, listener_factory=FakeListener, notify=lambda: None, state_dir=tmp_path
    )
    assert again.wake_enabled is False


def test_failed_start_returns_to_idle(tmp_path: Path) -> None:
    said: list[str] = []

    def broken() -> FakeSession:
        raise RuntimeError("no network")

    daemon = vc.VoiceDaemon(broken, announce=said.append, notify=lambda: None, state_dir=tmp_path)
    daemon.start_talking("test")
    assert daemon.state == "idle" and said == ["voice on", "voice failed"]


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
    reply = vc.send("status", path)
    assert reply is not None and reply["state"] == "idle"
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
    assert data["KeepAlive"] is True and data["RunAtLoad"] is True
    assert data["StandardErrorPath"] == str(tmp_path / "daemon.log")
