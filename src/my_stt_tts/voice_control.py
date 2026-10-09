"""mac-voice control daemon: one owner of the mic, toggled by wake word, chord or menu bar.

States: ``idle`` (the wake-word listener holds the mic, if wake is enabled), ``starting``,
``talking`` (an ElevenLabs :class:`~my_stt_tts.eleven_voice.VoiceSession` holds it) and
``stopping``. Every state change runs on ONE worker thread fed by a queue, so concurrent
chord / menu / wake / hang-up events are linearised.

Mic handoff (CoreAudio releases a device asynchronously): the wake listener is stopped,
its thread joined and a settle pause taken BEFORE a session opens the VoiceProcessingIO
engine; on the way back the session's audio is stopped and joined, "voice off" is spoken
and a cooldown passes BEFORE the listener re-arms — so neither the agent's voice nor the
announcements can fire the wake word, and ``idle`` is only published once that is done.

Control: a unix socket (``control.sock`` in the 0700 state dir) taking one-line commands
``on | off | toggle | status | wake on | wake off``; each reply is one JSON line. The
daemon writes ``status.json`` on every change (read by the SwiftBar plugin without
starting Python) and asks SwiftBar to refresh the plugin.

A session ends on ``off``/``toggle``, when the agent hangs up, after ``idle_timeout``
seconds without the user speaking while the agent is silent, or at ``max_duration``. If
a session cannot be reclaimed within ``stop_deadline`` the process exits non-zero so the
LaunchAgent restarts it with a clean audio stack.
"""

from __future__ import annotations

import contextlib
import functools
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
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

# Native/optional backends (sounddevice, openWakeWord) are imported lazily on purpose.
# pylint: disable=import-outside-toplevel

log = logging.getLogger("my_stt_tts.voice_control")

STATE_DIR = Path(os.environ.get("MAC_VOICE_STATE_DIR", Path.home() / ".local/state/mac-voice"))
IDLE_TIMEOUT_S = 60.0
MAX_DURATION_S = 15 * 60.0  # local backstop above the agent's own 10-minute cap
STOP_DEADLINE_S = 10.0
SETTLE_S = 0.3  # CoreAudio hands a released input device over asynchronously
REARM_COOLDOWN_S = 1.5  # room echo of "voice off" fades before the wake word listens
WAKE_PHRASE = "voice on"  # custom phrase (sherpa KWS); "hey jarvis" stays active too
WAKE_PHASES = 2  # see wake_listener_factory
WAKE_THRESHOLD = 0.75  # openWakeWord floor for the daemon (Albert scores 0.97+; noise 0.61)
SAMPLE_RATE = 16000
SWIFTBAR_REFRESH = (
    "swiftbar://refreshplugin?name=mac-voice",
    "swiftbar://refreshplugin?name=mac-voice.5s.sh",
)
LAUNCH_LABEL = "com.albert.mac-voice"
LAUNCH_AGENT = Path.home() / "Library/LaunchAgents" / f"{LAUNCH_LABEL}.plist"
REPO_ROOT = Path(__file__).resolve().parents[2]
READY_SOUND_ENV = "MAC_VOICE_READY_SOUND"
READY_SOUND = "/System/Library/Sounds/Glass.aiff"  # played once the daemon listens


class Session(Protocol):
    """What the daemon needs from a conversation (``eleven_voice.VoiceSession``)."""

    def start(self) -> None: ...
    def end(self) -> None: ...
    def wait(self) -> str | None: ...
    def idle_for(self) -> float: ...


class Listener(Protocol):
    """What the daemon needs from a wake-word listener (:class:`WakeListener`)."""

    def start(self) -> None: ...
    def stop(self) -> bool: ...


def socket_path(state_dir: Path = STATE_DIR) -> Path:
    return state_dir / "control.sock"


class WakeListener:
    """Run :func:`my_stt_tts.audio.listen_for_wake` on a thread; ``on_wake`` once, then stop.

    Reuses the main pipeline's wake loop (native-rate capture, resampling, exact 80 ms
    reframing, gain) instead of a second, simplified capture path.
    """

    def __init__(
        self,
        detector: Any,
        on_wake: Callable[[], None],
        *,
        gain: float = 1.0,
        gate: Any = None,
    ) -> None:
        self.detector = detector
        self.on_wake = on_wake
        self.gain = gain
        self.gate = gate  # voice_gate.VoiceGate: only enrolled voices may start
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="wake", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from . import audio

        fire_buffer = audio.WakeFireBuffer(SAMPLE_RATE, window_seconds=2.0)
        while not self._stop.is_set():
            try:
                fired = audio.listen_for_wake(
                    self.detector,
                    SAMPLE_RATE,
                    gain=self.gain,
                    stop=self._stop,
                    fire_buffer=fire_buffer,
                )
            except Exception:  # pylint: disable=broad-exception-caught
                log.exception("❌ wake listener crashed")
                return
            if not fired or self._stop.is_set():
                return
            if self._accept(fire_buffer.last_fire):
                self.on_wake()
                return

    def _accept(self, clip: Any) -> bool:
        """Log the fire; True when the voice behind it may start a conversation."""
        score = float(getattr(self.detector, "last_score", 0.0) or 0.0)
        if score >= float(getattr(self.detector, "threshold", 1.0)):
            what = f"{getattr(self.detector, 'model_name', '?')} (score {score:.2f})"
        else:  # the keyword spotter has no continuous score
            what = "custom phrase (keyword spotter)"
        if self.gate is None or not self.gate.active:
            log.info("🔔 wake: %s", what)
            return True
        ok, name, sim = self.gate.check(clip if clip is not None else [])
        if ok:
            log.info("🔔 wake: %s · 🗣️  %s (voice match %.2f)", what, name, sim)
        else:
            log.info(
                "🚫 wake ignored: %s · voice not enrolled (closest %s %.2f < %.2f)",
                what,
                name or "-",
                sim,
                self.gate.threshold,
            )
        return bool(ok)

    def stop(self) -> bool:
        """Stop and join; True once the capture thread (and its stream) is gone."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout=2.0)
        return not thread.is_alive()


def say(text: str) -> None:
    """Speak a short local announcement (macOS ``say``; blocks until spoken)."""
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(["/usr/bin/say", text], check=False, timeout=10)


def failure_announcement(session: Any) -> str:
    """What ``say`` announces for a failed start: "voice failed", plus why when known."""
    reason = str(getattr(session, "start_failure", "") or "").strip()
    return f"voice failed: {reason}" if reason else "voice failed"


def refresh_menu_bar() -> None:
    """Ask SwiftBar to re-run the mac-voice plugin now (no-op without SwiftBar)."""
    for url in SWIFTBAR_REFRESH:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            subprocess.run(
                ["/usr/bin/open", "-g", url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=5,
            )


class VoiceDaemon:  # pylint: disable=too-many-instance-attributes
    """State machine behind the control socket (see module docstring)."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        session_factory: Callable[[], Session],
        *,
        listener_factory: Callable[[Callable[[], None]], Listener] | None = None,
        announce: Callable[[str], None] = say,
        notify: Callable[[], None] = refresh_menu_bar,
        state_dir: Path = STATE_DIR,
        wake_enabled: bool | None = None,
        timing: dict[str, float] | None = None,
        bridge: Any = None,
    ) -> None:
        self.session_factory = session_factory
        self.bridge = bridge  # bridge.BridgeController (opt-in), shut down with the daemon
        self.listener_factory = listener_factory
        self.announce = announce
        self.notify = notify
        self.state_dir = state_dir
        self.timing = {
            "idle_timeout": IDLE_TIMEOUT_S,
            "max_duration": MAX_DURATION_S,
            "stop_deadline": STOP_DEADLINE_S,
            "settle": SETTLE_S,
            "cooldown": REARM_COOLDOWN_S,
        } | (timing or {})
        self.state = "idle"
        self._lock = threading.Lock()
        self._session: Session | None = None
        self._end_reason = ""
        self._listener: Listener | None = None
        self._events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._closed = threading.Event()
        prefs = self._read_prefs()
        default_wake = bool(prefs.get("wake", True)) if wake_enabled is None else wake_enabled
        self.wake_enabled = default_wake and listener_factory is not None
        threading.Thread(target=self._work, name="voice-worker", daemon=True).start()

    # -- persistence / publishing -------------------------------------------------------
    def _read_prefs(self) -> dict[str, Any]:
        try:
            data = json.loads((self.state_dir / "prefs.json").read_text())
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write(self, name: str, data: dict[str, Any]) -> None:
        self.state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
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

    def _set_state(self, state: str) -> None:
        with self._lock:
            self.state = state
        self.publish()

    def publish(self) -> None:
        with contextlib.suppress(OSError):
            self._write("status.json", self.status())
        self.notify()

    # -- the serialised worker ----------------------------------------------------------
    def submit(self, event: str, arg: Any = None) -> None:
        self._events.put((event, arg))

    def drain(self, timeout: float = 5.0) -> bool:
        """Block until every event queued so far has been processed (tests, shutdown)."""
        done = threading.Event()
        self.submit("mark", done)
        return done.wait(timeout)

    def _work(self) -> None:
        while not self._closed.is_set():
            try:
                event, arg = self._events.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._dispatch(event, arg)
            except Exception:  # pylint: disable=broad-exception-caught
                log.exception("❌ voice worker failed on %s", event)

    def _dispatch(self, event: str, arg: Any) -> None:
        if event == "on":
            self._start_talking(str(arg))
        elif event == "off":
            self._stop_talking()
        elif event == "toggle":
            if self.state == "idle":
                self._start_talking("toggle")
            else:
                self._stop_talking()
        elif event == "wake":
            self._set_wake(bool(arg))
        elif event == "arm":
            self._arm_wake()
        elif event == "finished":
            self._finish(arg)
        elif event == "mark":
            arg.set()

    # -- wake listener ------------------------------------------------------------------
    def _arm_wake(self) -> None:
        if not self.wake_enabled or self.state != "idle" or self._listener is not None:
            return
        assert self.listener_factory is not None
        listener = self.listener_factory(lambda: self.submit("on", "wake"))
        try:
            listener.start()
        except Exception:  # pylint: disable=broad-exception-caught
            log.exception("❌ wake listener failed to start")
            return
        self._listener = listener

    def _disarm_wake(self) -> bool:
        listener, self._listener = self._listener, None
        if listener is None:
            return True
        released = listener.stop()
        time.sleep(self.timing["settle"])
        return released

    def _set_wake(self, enabled: bool) -> None:
        with self._lock:
            self.wake_enabled = enabled and self.listener_factory is not None
        with contextlib.suppress(OSError):
            self._write("prefs.json", {"wake": enabled})
        if self.wake_enabled:
            self._arm_wake()
        else:
            self._disarm_wake()
        self.publish()

    # -- sessions -----------------------------------------------------------------------
    def _start_talking(self, reason: str) -> None:
        if self.state != "idle":
            return
        log.info("🟢 voice on — starting conversation (%s)", reason)
        self._set_state("starting")
        if not self._disarm_wake():
            log.error("❌ wake listener did not release the mic; not starting")
            self._set_state("idle")
            self._arm_wake()
            return
        self.announce("voice on")
        session: Session | None = None
        try:
            session = self.session_factory()
            session.start()
        except Exception:  # pylint: disable=broad-exception-caught
            log.exception("❌ conversation failed to start")
            self._set_state("stopping")
            self.announce(failure_announcement(session))
            self._back_to_idle()
            return
        self._session, self._end_reason = session, ""
        self._set_state("talking")
        threading.Thread(target=self._supervise, args=(session,), daemon=True).start()

    def _supervise(self, session: Session) -> None:
        done = threading.Event()

        def _wait() -> None:
            with contextlib.suppress(Exception):
                session.wait()
            done.set()

        threading.Thread(target=_wait, name="session-wait", daemon=True).start()
        started = time.monotonic()
        ended_at: float | None = None
        while not done.wait(0.25):
            if ended_at is not None:
                if time.monotonic() - ended_at > self.timing["stop_deadline"]:
                    self._fatal("session did not end within the stop deadline")
                    return
                continue
            if self.state == "stopping":  # a local "off" already asked it to end
                ended_at = time.monotonic()
                continue
            too_long = time.monotonic() - started > self.timing["max_duration"]
            try:
                idle = session.idle_for() > self.timing["idle_timeout"]
            except Exception:  # pylint: disable=broad-exception-caught
                log.warning("⚠️  idle check failed; treating the session as idle", exc_info=True)
                idle = True
            if too_long or idle:
                self._end_reason = "max duration" if too_long else "idle timeout"
                log.info("⏳ hanging up (%s)", self._end_reason)
                ended_at = time.monotonic()
                with contextlib.suppress(Exception):
                    session.end()
        self.submit("finished", session)
        report = getattr(session, "report", None)  # costs arrive seconds after the call
        if callable(report):
            threading.Thread(target=report, name="cost-report", daemon=True).start()

    def _fatal(self, why: str) -> None:
        """Unrecoverable audio/SDK state: exit so launchd restarts a clean process."""
        log.critical("❌ %s — exiting for a clean restart", why)
        with contextlib.suppress(OSError):
            (self.state_dir / "status.json").unlink()
        os._exit(70)  # pylint: disable=protected-access

    def _stop_talking(self) -> None:
        session = self._session
        if session is None or self.state != "talking":
            return
        self._set_state("stopping")
        self._end_reason = "you (vo / menu / off)"
        with contextlib.suppress(Exception):
            session.end()  # the supervisor sees wait() return and queues "finished"

    def _finish(self, session: Session) -> None:
        if self._session is not session:
            return
        self._session = None
        self._set_state("stopping")
        reason = (
            self._end_reason
            or getattr(session, "end_reason", "")
            or "remote: agent hung up or connection dropped (see costs)"
        )
        self._end_reason = ""
        failure = str(getattr(session, "start_failure", "") or "")
        if failure:  # the SDK died before the conversation opened (e.g. quota exceeded)
            log.info("🔴 voice off — the conversation never started (%s)", failure)
        else:
            log.info("🔴 voice off — conversation ended by %s", reason)
        time.sleep(self.timing["settle"])  # VoiceProcessingIO released before we speak
        self.announce(failure_announcement(session) if failure else "voice off")
        self._back_to_idle()

    def _back_to_idle(self) -> None:
        time.sleep(self.timing["cooldown"])  # tail of the announcement, then listen again
        self._set_state("idle")
        self._arm_wake()

    # -- commands -----------------------------------------------------------------------
    def handle(self, line: str) -> dict[str, Any]:
        cmd = " ".join(line.strip().lower().split())
        if cmd != "status":
            log.info("⌨️  command: %s (state %s)", cmd or "<empty>", self.state)
        if cmd in ("on", "off", "toggle"):
            self.submit(cmd, "command")
            time.sleep(0.05)  # usually lets the worker flip the state before replying
        elif cmd in ("wake on", "wake off"):
            self.submit("wake", cmd == "wake on")
            time.sleep(0.05)
        elif cmd != "status":
            return {"ok": False, "error": f"unknown command {cmd!r}", **self.status()}
        return {"ok": True, **self.status()}

    def shutdown(self) -> None:
        session = self._session
        if session is not None:
            with contextlib.suppress(Exception):
                session.end()
        listener, self._listener = self._listener, None
        if listener is not None:
            listener.stop()
        if self.bridge is not None:
            with contextlib.suppress(Exception):
                self.bridge.shutdown()
        self._closed.set()


def serve(daemon: VoiceDaemon, path: Path, stop: threading.Event) -> None:
    """Accept control connections on the unix socket ``path`` until ``stop`` is set."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if send("status", path, timeout=1.0) is not None:
            raise RuntimeError(f"another mac-voice daemon is already listening on {path}")
        path.unlink()
    # Bind under a temporary name, lock it down and listen, THEN rename into place: the
    # control socket only ever appears 0600 and already accepting connections.
    staging = path.with_name(f".{path.name}.{os.getpid()}")
    with contextlib.suppress(FileNotFoundError):
        staging.unlink()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(staging))
    staging.chmod(0o600)
    srv.listen(8)
    srv.settimeout(0.5)
    os.replace(staging, path)
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
        conn.settimeout(0.5)
        data = b""
        with contextlib.suppress(OSError):
            while b"\n" not in data and len(data) < 256:
                chunk = conn.recv(256)
                if not chunk:
                    break
                data += chunk
        reply = daemon.handle(data[:256].decode(errors="replace"))
        with contextlib.suppress(OSError):
            conn.sendall((json.dumps(reply) + "\n").encode())


def send(command: str, path: Path | None = None, timeout: float = 3.0) -> dict[str, Any] | None:
    """Send one command to the daemon; its JSON reply, or None when no daemon listens."""
    target = path or socket_path()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as cli:
            cli.settimeout(timeout)
            cli.connect(str(target))
            cli.sendall(command.encode() + b"\n")
            data = b""
            while not data.endswith(b"\n") and len(data) < 4096:
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


def wake_listener_factory(gate: Any = None) -> Callable[[Callable[[], None]], Listener] | None:
    """Build the configured wake detector once; None when openWakeWord is unavailable.

    ``gate`` reuses an already built :func:`voice_gate` (the bridge shares it).
    """
    try:
        from .config import Config
        from .wake import make_wake_detector

        cfg = Config.from_env(REPO_ROOT / ".env")
        if not Path(cfg.wake_model_path).is_absolute():
            cfg.wake_model_path = str(REPO_ROOT / cfg.wake_model_path)
        if not Path(cfg.kws_model_dir).is_absolute():
            cfg.kws_model_dir = str(REPO_ROOT / cfg.kws_model_dir)
        # The configured openWakeWord model ("hey jarvis") keeps working; the custom phrase
        # is OR'd in via sherpa KWS. Both are overridable per machine.
        cfg.wake_phrase = os.environ.get("MAC_VOICE_WAKE", WAKE_PHRASE)
        cfg.wake_threshold = float(os.environ.get("MAC_VOICE_WAKE_THRESHOLD", str(WAKE_THRESHOLD)))
        # 2 staggered openWakeWord copies instead of the pipeline's 8: ~12 % instead of
        # ~41 % of a core while idle, same hits on the test clips (measured 2026-10-09).
        cfg.wake_phases = int(os.environ.get("MAC_VOICE_WAKE_PHASES", str(WAKE_PHASES)))
        # The few-shot "enrolled" branch matched Albert's VOICE, not the phrase: it fired on
        # 34/43 recordings of his ordinary speech (2026-10-09) — never for the daemon.
        cfg.fewshot_wake_enabled = False
        detector = make_wake_detector(cfg)
        if not detector.available():
            return None
    except Exception:  # pylint: disable=broad-exception-caught
        log.warning("⚠️  wake word unavailable; chord + menu bar only", exc_info=True)
        return None
    gain = float(getattr(cfg, "wake_gain", 1.0))
    gate = gate if gate is not None else voice_gate()
    return lambda on_wake: WakeListener(detector, on_wake, gain=gain, gate=gate)


def voice_gate() -> Any:
    """The enrolled-voices check for wake fires (lets everyone in when nobody is enrolled)."""
    from .voice_gate import VoiceGate

    threshold = float(os.environ.get("MAC_VOICE_SPEAKER_THRESHOLD", "0.35"))
    gate = VoiceGate(threshold=threshold)
    if gate.active:
        log.info("🗣️  only enrolled voices start it: %s", ", ".join(gate.profiles))
    else:
        log.info("🗣️  no voice profiles — anyone can start it (mac-voice -V builds them)")
    if gate.call_profiles:
        log.info("🗣️  call-domain voice profiles: %s", ", ".join(gate.call_profiles))
    if gate.active or gate.call_profiles:
        gate.preload()
    return gate


def _build_bridge() -> tuple[Any, Any]:
    """(controller, voice gate) when a bridge flag is on, else (None, None)."""
    from .bridge import bridge_enabled, build_bridge

    if not bridge_enabled():
        return None, None
    gate = voice_gate()
    return build_bridge(gate), gate


def ready_sound(env: Mapping[str, str] | None = None) -> str | None:
    """The sound that says "listening" (``MAC_VOICE_READY_SOUND``); None when switched off.

    Unset → :data:`READY_SOUND`; set to ``0``, ``off`` or an empty string → no sound.
    """
    env = os.environ if env is None else env
    value = env.get(READY_SOUND_ENV)
    if value is None:
        return READY_SOUND
    value = value.strip()
    return None if value.casefold() in {"", "0", "off"} else value


def play_ready_sound(
    env: Mapping[str, str] | None = None,
    popen: Callable[..., Any] = subprocess.Popen,
) -> None:
    """Play the ready sound with ``afplay`` without waiting for it; failures only log."""
    sound = ready_sound(env)
    if sound is None:
        return
    try:
        popen(
            ["/usr/bin/afplay", sound],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:  # pylint: disable=broad-exception-caught  # a chime never matters
        log.debug("ready sound failed", exc_info=True)


def daemon_main(
    session_factory: Callable[..., Session],
    *,
    wake: bool = True,
    chime: Callable[[], None] = play_ready_sound,
) -> int:
    """Run the daemon in the foreground until SIGTERM / SIGINT.

    With a bridge flag on, the :class:`~my_stt_tts.bridge.BridgeController` is built here
    once and handed to every session as ``session_factory(bridge=controller)``. ``chime``
    plays once it listens (:func:`play_ready_sound`), so Albert knows he may say the wake
    word.
    """
    logging.getLogger("my_stt_tts").setLevel(logging.INFO)
    stop = threading.Event()
    bridge, gate = _build_bridge()
    if bridge is not None:
        session_factory = functools.partial(session_factory, bridge=bridge)
        log.info("🌉 Claude bridge on (authorised voice: %s)", bridge.authoriser.authorized or "-")
    listeners = (
        (wake_listener_factory(gate) if gate is not None else wake_listener_factory())
        if wake
        else None
    )
    daemon = VoiceDaemon(session_factory, listener_factory=listeners, bridge=bridge)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    daemon.submit("arm")
    daemon.publish()
    log.info("✅ mac-voice daemon listening on %s (wake: %s)", socket_path(), daemon.wake_enabled)
    try:
        chime()
    except Exception:  # pylint: disable=broad-exception-caught  # a chime never matters
        log.debug("ready chime failed", exc_info=True)
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
            "WorkingDirectory": str(launcher.parent),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ThrottleInterval": 10,
            "ProcessType": "Interactive",  # audio: no background throttling
            "StandardOutPath": log_file,
            "StandardErrorPath": log_file,
        }
    )


def install_launch_agent(launcher: Path) -> int:
    """Write + (re)load the LaunchAgent; the daemon then starts at every login."""
    domain = f"gui/{os.getuid()}"
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    LAUNCH_AGENT.parent.mkdir(parents=True, exist_ok=True)
    LAUNCH_AGENT.write_bytes(launch_agent_plist(launcher))
    subprocess.run(
        ["/bin/launchctl", "bootout", f"{domain}/{LAUNCH_LABEL}"], check=False, capture_output=True
    )
    res = subprocess.run(["/bin/launchctl", "bootstrap", domain, str(LAUNCH_AGENT)], check=False)
    ok = res.returncode == 0
    print(("✅ installed + started " if ok else "❌ launchctl failed for ") + str(LAUNCH_AGENT))
    return res.returncode


def uninstall_launch_agent() -> int:
    """Stop the LaunchAgent and remove its plist."""
    subprocess.run(
        ["/bin/launchctl", "bootout", f"gui/{os.getuid()}/{LAUNCH_LABEL}"],
        check=False,
        capture_output=True,
    )
    with contextlib.suppress(FileNotFoundError):
        LAUNCH_AGENT.unlink()
    print(f"✅ removed {LAUNCH_AGENT}")
    return 0
