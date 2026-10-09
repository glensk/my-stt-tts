"""Talk to an ElevenLabs agent from the Mac's own mic and speakers.

The ElevenLabs agent (speech-to-text, LLM, voice and turn-taking all run in their
cloud) is configured in the ElevenLabs dashboard; this module is only the local
ears + mouth: it streams 16 kHz PCM from the mic to the agent over the official
``elevenlabs`` SDK and plays the agent's audio back. Running the agent from a
local process (instead of the browser preview) is what later lets it call local
functions ("client tools"), e.g. to reach Claude Code sessions on this Mac.

Audio modes (``-m``):

* ``aec`` (default) — open speakers, full duplex: mic AND playback both run through
  one macOS VoiceProcessingIO engine (the FaceTime echo canceller,
  :class:`my_stt_tts.aec.VoiceProcessingDuplex`), so the agent never hears itself
  and you can talk over it — like the browser, which does the same with WebRTC.
  Uses the system default devices; falls back to ``speakers`` if it cannot start.
* ``speakers`` — the mic is muted while the agent speaks, so it never hears and
  interrupts itself; the price is that you cannot talk over it.
* ``headphones`` — raw mic + plain playback, full duplex; only safe with headphones.

Needs the ``elevenlabs`` + ``audio`` extras and ``ELEVENLABS_API_KEY`` /
``ELEVENLABS_AGENT_ID`` in the repo's ``.env``.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import logging
import os
import re
import signal
import sys
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

# Native/optional backends (sounddevice, elevenlabs, PyObjC) are imported lazily on purpose.
# pylint: disable=import-outside-toplevel

log = logging.getLogger("my_stt_tts.eleven_voice")

# "voice off" said to end the call, incl. how the recogniser hears it in German mode
# ("Weiß auf", "Weis aus") — matched locally so it never depends on the LLM's end_call.
VOICE_OFF = re.compile(
    r"^(?:voice|voiced|vois|voise|weiß|weiss|weis|wais|boys)\s*(?:off|of|auf|aus)$"
    r"|^stimme\s+aus$"
)
VOICE_RMS = 600.0  # int16 RMS (~-35 dBFS) above which the user counts as speaking
SAMPLE_RATE = 16000  # the SDK's fixed PCM format: 16-bit mono 16 kHz, both ways
INPUT_CHUNK = 4000  # 250 ms, the SDK's recommended input chunk
OUTPUT_BLOCK = 320  # 20 ms playback blocks keep interruption snappy
MODES = ("aec", "speakers", "headphones")


def _sd() -> Any:
    import sounddevice

    return sounddevice


class _Playback:
    """Byte buffer drained by a sounddevice output stream; clearable on interrupt."""

    def __init__(self, device: int | str | None) -> None:
        self._buf = bytearray()
        self._lock = threading.Lock()
        self.last_audio = 0.0
        self._stream = _sd().RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=OUTPUT_BLOCK,
            device=device,
            callback=self._callback,
        )

    def _callback(self, outdata: Any, _frames: int, _time: Any, _status: Any) -> None:
        n = len(outdata)
        with self._lock:
            chunk = bytes(self._buf[:n])
            del self._buf[:n]
        if chunk:
            self.last_audio = time.monotonic()
        outdata[: len(chunk)] = chunk
        outdata[len(chunk) :] = b"\x00" * (n - len(chunk))

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        self._stream.stop()
        self._stream.close()

    def push(self, audio: bytes) -> None:
        with self._lock:
            self._buf.extend(audio)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def speaking(self, tail_s: float = 0.3) -> bool:
        """True while audio is queued or was played within the last ``tail_s``."""
        with self._lock:
            pending = bool(self._buf)
        return pending or (time.monotonic() - self.last_audio) < tail_s


def make_audio_interface(  # pylint: disable=too-many-statements  # one nested class
    mode: str,
    in_dev: int | str | None,
    out_dev: int | str | None,
    on_frame: Callable[[np.ndarray], None] | None = None,
) -> Any:
    """Build an SDK ``AudioInterface`` for ``mode`` (see module docstring).

    ``on_frame`` (optional, the Claude bridge's turn source) also receives every mic frame
    sent to the agent, as 16 kHz float32. The class is nested so the optional
    ``elevenlabs`` base class is imported lazily.
    """
    from elevenlabs.conversational_ai.conversation import AudioInterface

    class MacAudioInterface(AudioInterface):  # pylint: disable=too-many-instance-attributes
        """VoiceProcessingIO duplex, or sounddevice capture + buffered playback."""

        def __init__(self) -> None:
            self.mode = mode
            self.playback = _Playback(out_dev)
            self._stop = threading.Event()
            self._in_stream: Any = None
            self._vp: Any = None
            self._thread: threading.Thread | None = None
            self._lifecycle = threading.Lock()  # start/stop never interleave
            self.last_user_audio = 0.0  # monotonic time the user was last audible

        def _send(self, cb: Callable[[bytes], None], pcm: bytes) -> None:
            if self.mode == "speakers" and self.playback.speaking():
                pcm = b"\x00" * len(pcm)  # gate the mic while the agent talks
            self._note_level(pcm)
            cb(pcm)
            if on_frame is not None:
                on_frame(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0)

        def _note_level(self, pcm: bytes) -> None:
            samples = np.frombuffer(pcm, dtype="<i2")
            if samples.size and np.sqrt(np.mean(samples.astype(np.float32) ** 2)) > VOICE_RMS:
                self.last_user_audio = time.monotonic()

        def start(self, input_callback: Callable[[bytes], None]) -> None:
            # The SDK opens audio only after the websocket connects; an end_session() in
            # between has already called stop(), so a late start must not open the mic.
            with self._lifecycle:
                if not self._stop.is_set():
                    self._open(input_callback)

        def _open(self, input_callback: Callable[[bytes], None]) -> None:
            if self.mode == "aec":
                if self._start_vp(input_callback):
                    return
                log.warning("⚠️  echo cancellation unavailable; mic muted while the agent talks.")
                self.mode = "speakers"
            self.playback.start()
            self._in_stream = _sd().RawInputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16",
                blocksize=INPUT_CHUNK,
                device=in_dev,
                callback=lambda indata, *_: self._send(input_callback, bytes(indata)),
            )
            self._in_stream.start()

        def _start_vp(self, input_callback: Callable[[bytes], None]) -> bool:
            from .aec import VoiceProcessingDuplex

            vp = VoiceProcessingDuplex(SAMPLE_RATE, frame_samples=INPUT_CHUNK)
            if not vp.start():
                return False
            self._vp = vp

            def _pump() -> None:
                for frame in vp.mic_frames():
                    if self._stop.is_set():
                        break
                    pcm = (np.clip(frame, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    self._note_level(pcm)
                    input_callback(pcm)
                    if on_frame is not None:
                        on_frame(frame)

            self._thread = threading.Thread(target=_pump, name="vp-capture", daemon=True)
            self._thread.start()
            return True

        def stop(self) -> None:
            with self._lifecycle:
                if self._stop.is_set():
                    return  # idempotent: the SDK and the daemon may both stop us
                self._stop.set()
                if self._in_stream is not None:
                    self._in_stream.stop()
                    self._in_stream.close()
                    self.playback.stop()
                if self._vp is not None:
                    self._vp.close()
                if self._thread is not None and self._thread is not threading.current_thread():
                    self._thread.join(timeout=2.0)  # capture thread gone before the mic is reused

        def output(self, audio: bytes) -> None:
            if self._vp is not None:
                self._vp.play(audio)
            else:
                self.playback.push(audio)

        def interrupt(self) -> None:
            if self._vp is not None:
                self._vp.flush()
            else:
                self.playback.clear()

        def speaking(self) -> bool:
            """True while agent audio is still queued or playing."""
            if self._vp is not None:
                return bool(self._vp.playing())
            return self.playback.speaking()

    return MacAudioInterface()


class EmojiFormatter(logging.Formatter):
    """Give every log line a leading symbol: keep the caller's emoji, else pick by level."""

    BY_LOGGER: ClassVar[dict[str, str]] = {"my_stt_tts.wake": "👂", "my_stt_tts.kws": "👂"}

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if msg[:1].isascii():  # no emoji of its own (e.g. the SDK's "Error receiving …")
            if record.levelno >= logging.ERROR:
                prefix = "❌"
            elif record.levelno >= logging.WARNING:
                prefix = "⚠️ "
            else:
                prefix = self.BY_LOGGER.get(record.name, "ℹ️ ")
            record = logging.makeLogRecord(record.__dict__ | {"msg": f"{prefix} {msg}", "args": ()})
        return super().format(record)


def is_voice_off(text: str) -> bool:
    """True when a user transcript is just the stop phrase ("Voice off.", "Weiß auf")."""
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return bool(VOICE_OFF.match(" ".join(words)))


def stamp(text: str) -> None:
    """Print one console line prefixed with the wall-clock time (HH:MM:SS)."""
    print(f"{time.strftime('%H:%M:%S')} {text}", flush=True)


def _load_env() -> None:
    """Load the repo's ``.env`` (two levels above ``src/my_stt_tts``) if present."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


#: Seconds the SDK gets to open the conversation (conversation id + first activity).
START_TIMEOUT_S = 10.0
#: A client tool never holds the SDK's thread pool longer than this (the tools self-limit).
TOOL_GUARD_S = 30.0
#: First word of a tool answer that may be logged (anything else is content → "answered").
_TOOL_OUTCOMES = frozenset(
    {"ok", "refused", "failed", "started", "confirmed", "needs", "nothing", "done", "cancelled"}
)


def _tool_outcome(reply: object) -> str:
    words = str(reply).split(maxsplit=1)
    head = words[0].rstrip(":,.").casefold() if words else ""
    return head if head in _TOOL_OUTCOMES else "answered"


def run_tool(name: str, fn: Callable[..., Any], args: dict[str, Any], guard: float) -> str:
    """Call ``fn(**args)`` on its own thread; a speakable string even on error or timeout.

    The SDK runs client tools in a thread pool without any timeout; this guard frees the
    pool slot after ``guard`` seconds (the tool's own subprocess limits end the worker).
    """
    box: dict[str, Any] = {}

    def _call() -> None:
        try:
            box["reply"] = fn(**args)
        except TypeError as exc:
            box["error"], box["bad_args"] = exc, True
        except Exception as exc:  # pylint: disable=broad-exception-caught  # tools never raise
            box["error"] = exc

    worker = threading.Thread(target=_call, name=f"tool-{name}", daemon=True)
    worker.start()
    worker.join(guard)
    if worker.is_alive():
        log.warning("⚠️  %s: no answer within %.0f s", name, guard)
        return f"{name} failed: it took too long"
    if "error" in box:
        if box.get("bad_args"):
            log.warning("⚠️  %s: bad arguments", name)
            return f"{name}: missing or unexpected arguments"
        log.warning("⚠️  %s failed (%s)", name, type(box["error"]).__name__)
        return f"{name} failed: internal error"
    reply = box.get("reply")
    return "done" if reply is None else str(reply)


class _SdkErrors(logging.Handler):
    """Keeps the SDK's last error line during startup (the close reason, e.g. 1008/3000)."""

    def __init__(self) -> None:
        super().__init__(logging.ERROR)
        self.last = ""

    def emit(self, record: logging.LogRecord) -> None:
        self.last = record.getMessage()[:300]


class VoiceSession:  # pylint: disable=too-many-instance-attributes  # one call's state
    """One ElevenLabs conversation: ``start`` returns at once, ``end`` hangs up, ``wait`` blocks.

    ``idle_for`` counts from the latest of: a transcript, the user's voice reaching the mic
    (before any transcript exists) and the end of agent playback — so a supervisor can hang
    up an idle session without cutting off a long utterance; the agent bills per minute.

    With a bridge (:class:`~my_stt_tts.bridge.BridgeController`) every controller tool is a
    client tool, the call carries ``[system notice]`` messages
    (:meth:`~my_stt_tts.bridge.BridgeController.notify_in_call`) and an open-problem
    briefing becomes the agent's first message. A start that dies silently (the SDK's
    receive thread ending before the conversation opened, e.g. ``3000 [quota_exceeded]``)
    sets :attr:`start_failure`, ends the session, posts a content-free banner and files an
    attention item — with or without a bridge.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        agent_id: str,
        api_key: str,
        *,
        mode: str = "aec",
        in_dev: str | None = None,
        out_dev: str | None = None,
        echo: bool = True,
        bridge: Any = None,
        start_timeout: float = START_TIMEOUT_S,
        tool_guard: float = TOOL_GUARD_S,
    ) -> None:
        from elevenlabs.client import ElevenLabs
        from elevenlabs.conversational_ai.conversation import (
            ClientTools,
            Conversation,
            ConversationInitiationData,
        )

        self.echo = echo
        self.agent_id, self._api_key = agent_id, api_key
        self.conversation_id: str | None = None
        self.end_reason = ""  # set when the session ends itself (e.g. "voice off" heard)
        self.start_failure = ""  # speakable reason when the conversation never opened
        self.last_activity = time.monotonic()
        self.bridge = bridge  # bridge.BridgeController when a bridge flag is on, else None
        self.start_timeout = start_timeout
        self.tool_guard = tool_guard
        self._transcript_seq = itertools.count(1)
        self._activity = threading.Event()  # an agent response or user transcript arrived
        self._ending = threading.Event()  # end() was asked for: not a startup failure
        self._start_error: BaseException | None = None
        self._sdk_errors = _SdkErrors()
        self._briefing_pending = False
        self.client_tools = ClientTools()
        override = self._briefing_override()
        on_frame = bridge.feed_audio if bridge is not None else None
        self.audio = make_audio_interface(mode, _device(in_dev), _device(out_dev), on_frame)
        self.conversation = Conversation(
            ElevenLabs(api_key=api_key),
            agent_id,
            requires_auth=True,
            audio_interface=self.audio,
            config=ConversationInitiationData(conversation_config_override=override or {}),
            client_tools=self.client_tools,
            callback_user_transcript=lambda t: self._said("🧑", t),
            callback_agent_response=lambda t: self._said("🤖", t),
            callback_latency_measurement=lambda ms: log.debug("latency %d ms", ms),
        )
        self._register_tools()

    # -- bridge: tools, briefing, notices ---------------------------------------------------
    def _register_tools(self) -> None:
        """Every controller tool becomes an SDK client tool (sync, parameters as a dict)."""
        if self.bridge is None:
            return
        for name, fn in sorted(self.bridge.tools.items()):
            self.client_tools.register(name, self._tool_handler(name, fn))

    def _tool_handler(self, name: str, fn: Callable[..., Any]) -> Callable[[dict], str]:
        bridge = self.bridge

        def handler(parameters: dict) -> str:
            args = {k: v for k, v in (parameters or {}).items() if k != "tool_call_id"}
            started = time.monotonic()
            bridge.tool_started(name)
            try:
                reply = run_tool(name, fn, args, self.tool_guard)
            finally:
                bridge.tool_finished(name)
                bridge.note_context("tool result")  # every tool result voids unused rights
            log.info("🛠️  %s → %s (%.1f s)", name, _tool_outcome(reply), time.monotonic() - started)
            return reply

        handler.__name__ = name
        return handler

    def _briefing_override(self) -> dict[str, Any] | None:
        """``{"agent": {"first_message": …}}`` when open problems wait to be briefed."""
        if self.bridge is None or not getattr(self.bridge, "first_message_ok", True):
            return None
        from .attention import first_message_override

        try:
            override = first_message_override(self.bridge.briefings)
        except Exception:  # pylint: disable=broad-exception-caught  # never block a call
            log.warning("⚠️  briefing unavailable", exc_info=True)
            return None
        self._briefing_pending = override is not None
        return override

    def _mark_briefed(self) -> None:
        if not self._briefing_pending:
            return
        self._briefing_pending = False
        mark = getattr(self.bridge.briefings, "mark_briefed", None)
        if callable(mark):
            try:
                mark()
                log.info("📣 open problems briefed")
            except Exception:  # pylint: disable=broad-exception-caught
                log.warning("⚠️  could not mark the briefing", exc_info=True)

    def _send_notice(self, text: str) -> None:
        self.conversation.send_user_message(text)

    # -- SDK callbacks --------------------------------------------------------------------
    def _said(self, who: str, text: str) -> None:
        self.last_activity = time.monotonic()
        self._activity.set()
        if who == "🤖" and self.bridge is not None:
            self._mark_briefed()  # the first agent turn is the briefing: it was spoken
        if who == "🧑" and self.bridge is not None:
            self._forward(text, self.last_activity)
        if who == "🧑" and is_voice_off(text) and not self.end_reason:
            self.end_reason = f"you said {text.strip()!r}"
            # end off the SDK's receive thread: end_session() tears that thread down
            threading.Thread(target=self.end, name="voice-off", daemon=True).start()
        if self.echo and text.strip(" .…"):  # "..." marks a silent turn, not speech
            stamp(f"{who} │ {text}")  # both emoji are 2 columns wide: transcripts line up

    def _forward(self, text: str, received_at: float) -> None:
        """Hand a user transcript to the bridge (never breaks the SDK's receive thread)."""
        if not text.strip(" .…"):
            return
        try:
            self.bridge.on_transcript(next(self._transcript_seq), text, received_at)
        except Exception:  # pylint: disable=broad-exception-caught
            log.warning("⚠️  bridge rejected a transcript", exc_info=True)

    # -- lifecycle --------------------------------------------------------------------------
    def start(self) -> None:
        if self.bridge is not None:
            self.bridge.begin_call()
        self._guard_sdk_thread()
        logging.getLogger("elevenlabs").addHandler(self._sdk_errors)
        try:
            self.conversation.start_session()
        except Exception as exc:
            self._detach_sdk_errors()
            self._startup_failed(exc)
            if self.bridge is not None:
                self.bridge.end_call()
            raise
        if self.bridge is not None:
            self.bridge.set_notice_sink(self._send_notice)
        threading.Thread(target=self._watch_start, name="voice-start", daemon=True).start()
        if self.echo:
            threading.Thread(target=self._print_models, name="models", daemon=True).start()

    def _guard_sdk_thread(self) -> None:
        """Catch what escapes the SDK's ``_run`` thread (it never calls end_session then)."""
        conversation = self.conversation
        sdk_run = conversation._run  # pylint: disable=protected-access

        def _run(ws_url: str) -> None:
            try:
                sdk_run(ws_url)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self._start_error = exc  # the watchdog turns it into a startup failure

        conversation._run = _run  # type: ignore[method-assign]  # pylint: disable=protected-access

    def _detach_sdk_errors(self) -> None:
        logging.getLogger("elevenlabs").removeHandler(self._sdk_errors)

    def _watch_start(self) -> None:
        """Until the call has opened: a dead SDK thread or no conversation id is a failure."""
        deadline = time.monotonic() + self.start_timeout
        try:
            while not self._ending.is_set() and not self._activity.is_set():
                thread = self.conversation._thread  # pylint: disable=protected-access
                dead = thread is not None and not thread.is_alive()
                if self._start_error is not None or dead:
                    if not self._activity.is_set() and not self._ending.is_set():
                        self._fail_start(self._start_error or self._sdk_errors.last)
                    return
                if time.monotonic() >= deadline:
                    if not self._conversation_open():
                        self._fail_start(f"no conversation within {self.start_timeout:.0f} s")
                    return
                time.sleep(0.05)
        finally:
            self._detach_sdk_errors()

    def _conversation_open(self) -> bool:
        return bool(self.conversation._conversation_id)  # pylint: disable=protected-access

    def _fail_start(self, cause: BaseException | str) -> None:
        self._startup_failed(cause)
        with contextlib.suppress(Exception):
            self.end()  # wait() returns at once: the daemon announces the failure

    def _startup_failed(self, cause: BaseException | str) -> None:
        """Log, banner and file a failed start; the briefing stays unbriefed (retried)."""
        from .attention import post_banner, report_startup_failure, startup_failure_reason

        reason = startup_failure_reason(cause or "")
        self._briefing_pending = False
        log.error("❌ ElevenLabs: %s", reason.removeprefix("ElevenLabs ").strip())
        if self.bridge is not None and "refused the start settings" in reason:
            if getattr(self.bridge, "first_message_ok", False):
                log.warning(
                    "⚠️  briefings off until restart: allow the first-message override "
                    "(scripts/eleven_agent_config.py -a)"
                )
            self.bridge.first_message_ok = False
        with contextlib.suppress(Exception):
            post_banner("voice startup failed")
        if self.bridge is not None:
            self._file_startup_failure(reason, report_startup_failure)
        self.start_failure = reason  # last: whoever sees it sees a fully handled failure

    def _file_startup_failure(self, reason: str, report: Callable[..., Any]) -> None:
        store = self.bridge.problems
        try:
            if hasattr(store, "record"):  # the attention inbox
                report(store, reason)
            else:
                from .bridge import Problem

                store.report(Problem("voice_startup_failed", "voice agent", reason))
        except Exception:  # pylint: disable=broad-exception-caught
            log.warning("⚠️  could not file the startup failure", exc_info=True)

    def end(self) -> None:
        self._ending.set()
        self.conversation.end_session()

    def wait(self) -> str | None:
        try:
            self.conversation_id = self.conversation.wait_for_session_end()
        finally:
            if self.bridge is not None:
                self.bridge.set_notice_sink(None)
                self.bridge.end_call()
        return self.conversation_id

    def _print_models(self) -> None:
        from .eleven_costs import agent_models

        try:
            stamp(agent_models(self._api_key, self.agent_id))
        except Exception:  # pylint: disable=broad-exception-caught  # display only
            log.debug("model lookup failed", exc_info=True)

    def report(self) -> None:
        """Print this conversation's models + costs once ElevenLabs has finalised it."""
        from .eleven_costs import fetch_report

        if not self.conversation_id:
            return
        try:
            for line in fetch_report(self._api_key, self.conversation_id):
                stamp(line)
            print(flush=True)  # blank line: the cost summary closes a session's block
        except Exception:  # pylint: disable=broad-exception-caught  # display only
            log.warning("⚠️  cost report failed", exc_info=True)

    def idle_for(self) -> float:
        """Seconds since the user or agent last did anything (0 while the agent speaks)."""
        if self.audio.speaking():
            self.last_activity = time.monotonic()
        latest = max(self.last_activity, self.audio.last_user_audio)
        return time.monotonic() - latest


def credentials(agent_id: str | None = None) -> tuple[str, str] | None:
    """``(agent_id, api_key)`` from the arguments / repo ``.env``, or None when missing."""
    _load_env()
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    agent = agent_id or os.environ.get("ELEVENLABS_AGENT_ID")
    if not api_key or not agent:
        return None
    return agent, api_key


def run(agent_id: str, api_key: str, mode: str, in_dev: str | None, out_dev: str | None) -> int:
    """Run one conversation until Ctrl-C or until the agent ends it."""
    session = VoiceSession(agent_id, api_key, mode=mode, in_dev=in_dev, out_dev=out_dev)
    signal.signal(signal.SIGINT, lambda *_: session.end())
    stamp(f"✅ connected ({mode}) — talk now; Ctrl-C ends the conversation.")
    session.start()
    conversation_id = session.wait()
    stamp(f"✅ conversation ended (id {conversation_id}).")
    session.report()
    return 0


def _control_command(args: argparse.Namespace) -> str | None:
    """The daemon command a client flag asks for, or None."""
    for flag, command in (("toggle", "toggle"), ("on", "on"), ("off", "off"), ("status", "status")):
        if getattr(args, flag):
            return command
    return f"wake {args.wake}" if args.wake else None


def _send_control(command: str) -> int:
    from .voice_control import send, socket_path

    reply = send(command)
    if reply is None:
        print(f"❌ no mac-voice daemon on {socket_path()} — start it with: mac-voice -d")
        return 1
    print(json.dumps(reply))
    return 0 if reply.get("ok") else 1


def _admin(args: argparse.Namespace) -> int | None:
    """Handle LaunchAgent, voice-profile and daemon-control flags; None to go on."""
    if args.voices:
        return _rebuild_voices()
    if args.doctor:
        from .mac_control import doctor_main

        return doctor_main()
    if args.install or args.uninstall:
        from . import voice_control

        if args.uninstall:
            return voice_control.uninstall_launch_agent()
        return voice_control.install_launch_agent(Path(__file__).resolve().parents[2] / "mac-voice")
    command = _control_command(args)
    return None if command is None else _send_control(command)


def _rebuild_voices() -> int:
    from .voice_gate import build_profile, enrolled_speakers

    names = enrolled_speakers()
    if not names:
        print("❌ no enrollment clips — record some: scripts/enroll_wakeword.py 'voice on' -w NAME")
        return 1
    for who in names:
        path, used, found = build_profile(who)
        if path is None:
            print(f"⚠️  {who}: only {found} clips — need at least 3")
        else:
            print(f"✅ {who}: voice profile from {used}/{found} clips → {path}")
    print("Restart the daemon to use them (only these voices can start a conversation).")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point (``mac-voice``)."""
    parser = argparse.ArgumentParser(
        prog="mac-voice",
        description="Talk to your ElevenLabs agent through this Mac's mic and speakers.",
        epilog=(
            "examples:\n"
            "  mac-voice                 open speakers + macOS echo cancellation (talk over it)\n"
            "  mac-voice -m speakers     open speakers, mic muted while the agent talks\n"
            "  mac-voice -m headphones   raw mic + plain playback (headphones only)\n"
            "  mac-voice -l              list audio devices\n"
            "  mac-voice -i 2 -o 3       pick input/output device by index or name\n"
            "\n"
            "background daemon (wake word + chord + menu bar):\n"
            "  mac-voice -d              run the daemon in the foreground\n"
            "  mac-voice -t              toggle the conversation (Karabiner v+o)\n"
            "  mac-voice -n / -f         conversation on / off\n"
            "  mac-voice -w off          disable the wake word (on: enable)\n"
            "  mac-voice -s              daemon status as JSON\n"
            "  mac-voice -I / -U         install / remove the login LaunchAgent\n"
            "  mac-voice -V              rebuild voice profiles (only enrolled voices may start)\n"
            "  mac-voice -D              doctor: Mac permissions for Mac control (✅/❌ + fix)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--mode", choices=MODES, default="aec", help="audio mode")
    parser.add_argument("-a", "--agent-id", help="agent id (default: ELEVENLABS_AGENT_ID)")
    parser.add_argument("-i", "--input-device", help="input device (speakers/headphones modes)")
    parser.add_argument("-o", "--output-device", help="output device (speakers/headphones modes)")
    parser.add_argument("-l", "--list-devices", action="store_true", help="list audio devices")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging (latency)")
    daemon = parser.add_argument_group("daemon")
    ctl = daemon.add_mutually_exclusive_group()
    ctl.add_argument("-d", "--daemon", action="store_true", help="run the control daemon")
    ctl.add_argument("-t", "--toggle", action="store_true", help="toggle the conversation")
    ctl.add_argument("-n", "--on", action="store_true", help="start a conversation")
    ctl.add_argument("-f", "--off", action="store_true", help="end the conversation")
    ctl.add_argument("-s", "--status", action="store_true", help="print daemon status")
    ctl.add_argument("-w", "--wake", choices=("on", "off"), help="enable/disable the wake word")
    ctl.add_argument(
        "-V", "--voices", action="store_true", help="(re)build voice profiles from enroll clips"
    )
    ctl.add_argument("-I", "--install", action="store_true", help="install the login LaunchAgent")
    ctl.add_argument("-U", "--uninstall", action="store_true", help="remove the LaunchAgent")
    ctl.add_argument(
        "-D", "--doctor", action="store_true", help="check the Mac permissions of Mac control"
    )
    daemon.add_argument("-W", "--no-wake", action="store_true", help="daemon without wake word")
    args = parser.parse_args(argv)

    handler = logging.StreamHandler()
    handler.setFormatter(EmojiFormatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    # speechbrain sets up its own loggers and announces every model file it loads
    handler.addFilter(
        lambda r: r.levelno >= logging.WARNING or not r.name.startswith("speechbrain")
    )
    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")  # onnxruntime on macOS
    log.setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    if args.list_devices:
        print(_sd().query_devices())
        return 0

    handled = _admin(args)
    if handled is not None:
        return handled
    creds = credentials(args.agent_id)
    if creds is None:
        print("❌ ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID must be set (repo .env).")
        return 2
    try:
        if args.daemon:
            from .voice_control import daemon_main

            def factory(bridge: Any = None) -> VoiceSession:
                return VoiceSession(
                    *creds,
                    mode=args.mode,
                    in_dev=args.input_device,
                    out_dev=args.output_device,
                    bridge=bridge,
                )

            return daemon_main(factory, wake=not args.no_wake)
        return run(*creds, args.mode, args.input_device, args.output_device)
    except ImportError as exc:
        print(f"❌ missing dependency ({exc.name}); run: uv sync --inexact --extra elevenlabs")
        return 3


if __name__ == "__main__":
    sys.exit(main())
