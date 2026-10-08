"""mac-voice control daemon: one owner of the mic, toggled by wake word, chord or menu bar.

States: ``idle`` (the wake-word listener holds the mic, if wake is enabled), ``starting``
and ``talking`` (an ElevenLabs :class:`~my_stt_tts.eleven_voice.VoiceSession` holds it).
The wake listener is always closed before a session opens the VoiceProcessingIO engine and
re-armed only after the session is gone and "voice off" was spoken, so the agent's voice
and the announcements can never fire the wake word.

Control: a unix socket (``control.sock`` in the state dir) taking one-line commands
``on | off | toggle | status | wake on | wake off``; each reply is one JSON line. The
daemon also writes ``status.json`` there on every change (read by the SwiftBar plugin
without starting Python) and asks SwiftBar to refresh the ``mac-voice`` plugin.

A session ends on ``off``/``toggle``, when the agent hangs up, or after ``idle_timeout``
seconds without a transcript while the agent is silent (the agent is billed per
connected minute).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import plistlib
import queue
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import numpy as np

# Native/optional backends (sounddevice, openWakeWord) are imported lazily on purpose.
# pylint: disable=import-outside-toplevel

log = logging.getLogger("my_stt_tts.voice_control")

STATE_DIR = Path(os.environ.get("MAC_VOICE_STATE_DIR", Path.home() / ".local/state/mac-voice"))
IDLE_TIMEOUT_S = 60.0
WAKE_FRAME = 1280  # 80 ms at 16 kHz, openWakeWord's frame
SAMPLE_RATE = 16000
SWIFTBAR_REFRESH = "swiftbar://refreshplugin?name=mac-voice"
LAUNCH_LABEL = "com.albert.mac-voice"
LAUNCH_AGENT = Path.home() / "Library/LaunchAgents" / f"{LAUNCH_LABEL}.plist"


class Session(Protocol):
    """What the daemon needs from a conversation (``eleven_voice.VoiceSession``)."""

    def start(self) -> None: ...
    def end(self) -> None: ...
    def wait(self) -> str | None: ...
    def idle_for(self) -> float: ...


class Listener(Protocol):
    """What the daemon needs from a wake-word listener (:class:`WakeListener`)."""

    def start(self) -> None: ...
    def stop(self) -> None: ...


def socket_path(state_dir: Path = STATE_DIR) -> Path:
    return state_dir / "control.sock"


class WakeListener:
    """Mic -> wake detector on a background thread; calls ``on_wake`` once, then stops."""

    def __init__(self, detector: Any, on_wake: Callable[[], None]) -> None:
        self.detector = detector
        self.on_wake = on_wake
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stop = threading.Event()
        self._stream: Any = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        import sounddevice as sd

        self._stop.clear()
        with contextlib.suppress(Exception):
            self.detector.reset()
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=WAKE_FRAME,
            callback=self._on_audio,
        )
        self._stream.start()
        self._thread = threading.Thread(target=self._loop, name="wake", daemon=True)
        self._thread.start()

    def _on_audio(self, indata: Any, _frames: int, _time: Any, _status: Any) -> None:
        with contextlib.suppress(queue.Full):
            self._q.put_nowait(indata[:, 0].copy())

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.detector.detect(frame):
                log.info("wake word fired (score %.2f)", getattr(self.detector, "last_score", 0))
                self._stop.set()
                threading.Thread(target=self.on_wake, name="wake-fire", daemon=True).start()
                return

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            with contextlib.suppress(Exception):
                self._stream.stop()
                self._stream.close()
            self._stream = None
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)
        self._thread = None


def say(text: str) -> None:
    """Speak a short local announcement (macOS ``say``; blocks until spoken)."""
    with contextlib.suppress(OSError):
        subprocess.run(["say", text], check=False, timeout=10)


def refresh_menu_bar() -> None:
    """Ask SwiftBar to re-run the mac-voice plugin now (no-op without SwiftBar)."""
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(
            ["open", "-g", SWIFTBAR_REFRESH],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )


class VoiceDaemon:  # pylint: disable=too-many-instance-attributes
    """State machine behind the control socket (see module docstring)."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        listener_factory: Callable[[Callable[[], None]], Listener] | None = None,
        announce: Callable[[str], None] = say,
        notify: Callable[[], None] = refresh_menu_bar,
        idle_timeout: float = IDLE_TIMEOUT_S,
        state_dir: Path = STATE_DIR,
        wake_enabled: bool | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.listener_factory = listener_factory
        self.announce = announce
        self.notify = notify
        self.idle_timeout = idle_timeout
        self.state_dir = state_dir
        self.state = "idle"
        self._lock = threading.RLock()
        self._session: Session | None = None
        self._listener: Listener | None = None
        self._cancel = False
        prefs = self._read_prefs()
        default_wake = bool(prefs.get("wake", True)) if wake_enabled is None else wake_enabled
        self.wake_enabled = default_wake and listener_factory is not None

    # -- persistence --------------------------------------------------------------------
    def _read_prefs(self) -> dict[str, Any]:
        try:
            data = json.loads((self.state_dir / "prefs.json").read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, name: str, data: dict[str, Any]) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_dir / f".{name}.tmp"
        tmp.write_text(json.dumps(data))
        tmp.replace(self.state_dir / name)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self.state,
                "wake": self.wake_enabled,
                "wake_available": self.listener_factory is not None,
                "pid": os.getpid(),
            }

    def _changed(self) -> None:
        with contextlib.suppress(OSError):
            self._write("status.json", self.status())
        self.notify()

    # -- wake listener ------------------------------------------------------------------
    def arm_wake(self) -> None:
        with self._lock:
            if not self.wake_enabled or self.state != "idle" or self._listener is not None:
                return
            assert self.listener_factory is not None
            listener = self.listener_factory(lambda: self.start_talking("wake"))
            try:
                listener.start()
            except Exception:  # pylint: disable=broad-exception-caught
                log.exception("❌ wake listener failed to start")
                return
            self._listener = listener

    def disarm_wake(self) -> None:
        with self._lock:
            listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()

    def set_wake(self, enabled: bool) -> None:
        with self._lock:
            self.wake_enabled = enabled and self.listener_factory is not None
        with contextlib.suppress(OSError):
            self._write("prefs.json", {"wake": enabled})
        if self.wake_enabled:
            self.arm_wake()
        else:
            self.disarm_wake()
        self._changed()

    # -- sessions -----------------------------------------------------------------------
    def start_talking(self, reason: str) -> None:
        with self._lock:
            if self.state != "idle":
                return
            self.state, self._cancel = "starting", False
        log.info("starting conversation (%s)", reason)
        self._changed()
        self.disarm_wake()  # release the mic before VoiceProcessingIO takes it
        self.announce("voice on")
        try:
            session = self.session_factory()
            session.start()
        except Exception:  # pylint: disable=broad-exception-caught
            log.exception("❌ conversation failed to start")
            with self._lock:
                self.state = "idle"
            self.announce("voice failed")
            self.arm_wake()
            self._changed()
            return
        with self._lock:
            self._session, self.state = session, "talking"
            cancel = self._cancel
        self._changed()
        threading.Thread(target=self._supervise, args=(session,), daemon=True).start()
        if cancel:
            session.end()

    def _supervise(self, session: Session) -> None:
        done = threading.Event()

        def _wait() -> None:
            with contextlib.suppress(Exception):
                session.wait()
            done.set()

        threading.Thread(target=_wait, name="session-wait", daemon=True).start()
        ending = False
        while not done.wait(0.5):
            if not ending and session.idle_for() > self.idle_timeout:
                log.info("idle for %.0f s — hanging up", self.idle_timeout)
                ending = True
                session.end()
        with self._lock:
            if self._session is session:
                self._session, self.state = None, "idle"
        self.announce("voice off")
        self.arm_wake()
        self._changed()

    def stop_talking(self) -> None:
        with self._lock:
            session = self._session
            if self.state == "starting":
                self._cancel = True
        if session is not None:
            session.end()

    # -- commands -----------------------------------------------------------------------
    def handle(self, line: str) -> dict[str, Any]:
        cmd = " ".join(line.strip().lower().split())
        if cmd in ("on", "toggle") and (cmd == "on" or self.state == "idle"):
            threading.Thread(target=self.start_talking, args=(cmd,), daemon=True).start()
            time.sleep(0.05)  # let the state flip to "starting" before replying
        elif cmd in ("off", "toggle"):
            self.stop_talking()
        elif cmd in ("wake on", "wake off"):
            self.set_wake(cmd == "wake on")
        elif cmd != "status":
            return {"ok": False, "error": f"unknown command {cmd!r}", **self.status()}
        return {"ok": True, **self.status()}

    def shutdown(self) -> None:
        self.disarm_wake()
        self.stop_talking()


def serve(daemon: VoiceDaemon, path: Path, stop: threading.Event) -> None:
    """Accept control connections on the unix socket ``path`` until ``stop`` is set."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if send("status", path) is not None:
            raise RuntimeError(f"another mac-voice daemon is already listening on {path}")
        path.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(path))
    path.chmod(0o600)
    srv.listen(8)
    srv.settimeout(0.5)
    try:
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            threading.Thread(target=_answer, args=(daemon, conn), daemon=True).start()
    finally:
        srv.close()
        with contextlib.suppress(OSError):
            path.unlink()


def _answer(daemon: VoiceDaemon, conn: socket.socket) -> None:
    with conn:
        conn.settimeout(2.0)
        data = b""
        with contextlib.suppress(OSError):
            while b"\n" not in data and len(data) < 256:
                chunk = conn.recv(256)
                if not chunk:
                    break
                data += chunk
        reply = daemon.handle(data.decode(errors="replace"))
        with contextlib.suppress(OSError):
            conn.sendall((json.dumps(reply) + "\n").encode())


def send(command: str, path: Path | None = None, timeout: float = 5.0) -> dict[str, Any] | None:
    """Send one command to the daemon; its JSON reply, or None when no daemon listens."""
    target = path or socket_path()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as cli:
            cli.settimeout(timeout)
            cli.connect(str(target))
            cli.sendall(command.encode() + b"\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = cli.recv(4096)
                if not chunk:
                    break
                data += chunk
    except OSError:
        return None
    try:
        reply = json.loads(data)
    except ValueError:
        return None
    return reply if isinstance(reply, dict) else None


def wake_listener_factory() -> Callable[[Callable[[], None]], Listener] | None:
    """Build the configured wake detector once; None when openWakeWord is unavailable."""
    try:
        from .config import Config
        from .wake import make_wake_detector

        detector = make_wake_detector(Config.from_env())
        if not detector.available():
            return None
    except Exception:  # pylint: disable=broad-exception-caught
        log.warning("⚠️  wake word unavailable; chord + menu bar only", exc_info=True)
        return None
    return lambda on_wake: WakeListener(detector, on_wake)


def daemon_main(session_factory: Callable[[], Session], *, wake: bool = True) -> int:
    """Run the daemon in the foreground until SIGTERM / SIGINT."""
    stop = threading.Event()
    daemon = VoiceDaemon(
        session_factory, listener_factory=wake_listener_factory() if wake else None
    )
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    daemon.arm_wake()
    daemon._changed()  # pylint: disable=protected-access  # publish the initial status
    print(f"✅ mac-voice daemon listening on {socket_path()} (wake: {daemon.wake_enabled})")
    try:
        serve(daemon, socket_path(), stop)
    finally:
        daemon.shutdown()
        with contextlib.suppress(OSError):
            (STATE_DIR / "status.json").unlink()
        refresh_menu_bar()
    return 0


def launch_agent_plist(launcher: Path, state_dir: Path = STATE_DIR) -> bytes:
    """The LaunchAgent that keeps ``<launcher> -d`` running for this user."""
    log_file = str(state_dir / "daemon.log")
    return plistlib.dumps(
        {
            "Label": LAUNCH_LABEL,
            "ProgramArguments": [str(launcher), "-d"],
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Interactive",  # audio: no background throttling
            "StandardOutPath": log_file,
            "StandardErrorPath": log_file,
        }
    )


def install_launch_agent(launcher: Path) -> int:
    """Write + (re)load the LaunchAgent; the daemon then starts at every login."""
    domain = f"gui/{os.getuid()}"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
    LAUNCH_AGENT.write_bytes(launch_agent_plist(launcher))
    subprocess.run(
        ["launchctl", "bootout", f"{domain}/{LAUNCH_LABEL}"], check=False, capture_output=True
    )
    res = subprocess.run(["launchctl", "bootstrap", domain, str(LAUNCH_AGENT)], check=False)
    print(
        ("✅ installed + started " if res.returncode == 0 else "❌ launchctl failed for ")
        + str(LAUNCH_AGENT)
    )
    return res.returncode


def uninstall_launch_agent() -> int:
    """Stop the LaunchAgent and remove its plist."""
    subprocess.run(
        ["launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCH_LABEL}"],
        check=False,
        capture_output=True,
    )
    with contextlib.suppress(FileNotFoundError):
        LAUNCH_AGENT.unlink()
    print(f"✅ removed {LAUNCH_AGENT}")
    return 0
