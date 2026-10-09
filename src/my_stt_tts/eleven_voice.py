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
import json
import logging
import os
import signal
import sys
import threading
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Native/optional backends (sounddevice, elevenlabs, PyObjC) are imported lazily on purpose.
# pylint: disable=import-outside-toplevel

log = logging.getLogger("my_stt_tts.eleven_voice")

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
    mode: str, in_dev: int | str | None, out_dev: int | str | None
) -> Any:
    """Build an SDK ``AudioInterface`` for ``mode`` (see module docstring).

    The class is nested so the optional ``elevenlabs`` base class is imported lazily.
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


class VoiceSession:
    """One ElevenLabs conversation: ``start`` returns at once, ``end`` hangs up, ``wait`` blocks.

    ``idle_for`` counts from the latest of: a transcript, the user's voice reaching the mic
    (before any transcript exists) and the end of agent playback — so a supervisor can hang
    up an idle session without cutting off a long utterance; the agent bills per minute.
    """

    def __init__(
        self,
        agent_id: str,
        api_key: str,
        *,
        mode: str = "aec",
        in_dev: str | None = None,
        out_dev: str | None = None,
        echo: bool = True,
    ) -> None:
        from elevenlabs.client import ElevenLabs
        from elevenlabs.conversational_ai.conversation import ClientTools, Conversation

        self.echo = echo
        self.agent_id, self._api_key = agent_id, api_key
        self.conversation_id: str | None = None
        self.last_activity = time.monotonic()
        self.audio = make_audio_interface(mode, _device(in_dev), _device(out_dev))
        self.conversation = Conversation(
            ElevenLabs(api_key=api_key),
            agent_id,
            requires_auth=True,
            audio_interface=self.audio,
            client_tools=ClientTools(),  # local functions get registered here (Claude Code bridge)
            callback_user_transcript=lambda t: self._said("you", t),
            callback_agent_response=lambda t: self._said("agent", t),
            callback_latency_measurement=lambda ms: log.debug("latency %d ms", ms),
        )

    def _said(self, who: str, text: str) -> None:
        self.last_activity = time.monotonic()
        if self.echo and text.strip(" .…"):  # "..." marks a silent turn, not speech
            stamp(f"{who:<5} │ {text}")  # fixed-width label: both transcripts line up

    def start(self) -> None:
        self.conversation.start_session()
        if self.echo:
            threading.Thread(target=self._print_models, name="models", daemon=True).start()

    def end(self) -> None:
        self.conversation.end_session()

    def wait(self) -> str | None:
        self.conversation_id = self.conversation.wait_for_session_end()
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
    """Handle LaunchAgent and daemon-control flags; None when the CLI should go on."""
    if args.install or args.uninstall:
        from . import voice_control

        if args.uninstall:
            return voice_control.uninstall_launch_agent()
        return voice_control.install_launch_agent(Path(__file__).resolve().parents[2] / "mac-voice")
    command = _control_command(args)
    return None if command is None else _send_control(command)


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
            "  mac-voice -I / -U         install / remove the login LaunchAgent"
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
    ctl.add_argument("-I", "--install", action="store_true", help="install the login LaunchAgent")
    ctl.add_argument("-U", "--uninstall", action="store_true", help="remove the LaunchAgent")
    daemon.add_argument("-W", "--no-wake", action="store_true", help="daemon without wake word")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
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

            def factory() -> VoiceSession:
                return VoiceSession(
                    *creds, mode=args.mode, in_dev=args.input_device, out_dev=args.output_device
                )

            return daemon_main(factory, wake=not args.no_wake)
        return run(*creds, args.mode, args.input_device, args.output_device)
    except ImportError as exc:
        print(f"❌ missing dependency ({exc.name}); run: uv sync --inexact --extra elevenlabs")
        return 3


if __name__ == "__main__":
    sys.exit(main())
