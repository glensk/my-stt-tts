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
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# Native/optional backends (sounddevice, elevenlabs, PyObjC) are imported lazily on purpose.
# pylint: disable=import-outside-toplevel

log = logging.getLogger("my_stt_tts.eleven_voice")

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


def make_audio_interface(mode: str, in_dev: int | str | None, out_dev: int | str | None) -> Any:
    """Build an SDK ``AudioInterface`` for ``mode`` (see module docstring)."""
    from elevenlabs.conversational_ai.conversation import AudioInterface

    class MacAudioInterface(AudioInterface):
        """VoiceProcessingIO duplex, or sounddevice capture + buffered playback."""

        def __init__(self) -> None:
            self.mode = mode
            self.playback = _Playback(out_dev)
            self._stop = threading.Event()
            self._in_stream: Any = None
            self._vp: Any = None
            self._thread: threading.Thread | None = None

        def _send(self, cb: Callable[[bytes], None], pcm: bytes) -> None:
            if self.mode == "speakers" and self.playback.speaking():
                pcm = b"\x00" * len(pcm)  # gate the mic while the agent talks
            cb(pcm)

        def start(self, input_callback: Callable[[bytes], None]) -> None:
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
                    input_callback(pcm)

            self._thread = threading.Thread(target=_pump, name="vp-capture", daemon=True)
            self._thread.start()
            return True

        def stop(self) -> None:
            self._stop.set()
            if self._in_stream is not None:
                self._in_stream.stop()
                self._in_stream.close()
                self.playback.stop()
            if self._vp is not None:
                self._vp.close()

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

    return MacAudioInterface()


def _load_env() -> None:
    """Load the repo's ``.env`` (two levels above ``src/my_stt_tts``) if present."""
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


def _device(value: str | None) -> int | str | None:
    if value is None:
        return None
    return int(value) if value.isdigit() else value


def run(agent_id: str, api_key: str, mode: str, in_dev: str | None, out_dev: str | None) -> int:
    """Run one conversation until Ctrl-C or until the agent ends it."""
    from elevenlabs.client import ElevenLabs
    from elevenlabs.conversational_ai.conversation import (
        ClientTools,
        Conversation,
    )

    conversation = Conversation(
        ElevenLabs(api_key=api_key),
        agent_id,
        requires_auth=True,
        audio_interface=make_audio_interface(mode, _device(in_dev), _device(out_dev)),
        client_tools=ClientTools(),  # local functions get registered here (Claude Code bridge)
        callback_user_transcript=lambda t: print(f"🎙️  you:   {t}", flush=True),
        callback_agent_response=lambda t: print(f"🤖 agent: {t}", flush=True),
        callback_latency_measurement=lambda ms: log.debug("latency %d ms", ms),
    )
    signal.signal(signal.SIGINT, lambda *_: conversation.end_session())
    print(f"✅ connected ({mode}) — talk now; Ctrl-C ends the conversation.", flush=True)
    conversation.start_session()
    conversation_id = conversation.wait_for_session_end()
    print(f"✅ conversation ended (id {conversation_id}).", flush=True)
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
            "  mac-voice -i 2 -o 3       pick input/output device by index or name"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-m", "--mode", choices=MODES, default="aec", help="audio mode")
    parser.add_argument("-a", "--agent-id", help="agent id (default: ELEVENLABS_AGENT_ID)")
    parser.add_argument("-i", "--input-device", help="input device (speakers/headphones modes)")
    parser.add_argument("-o", "--output-device", help="output device (speakers/headphones modes)")
    parser.add_argument("-l", "--list-devices", action="store_true", help="list audio devices")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging (latency)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    log.setLevel(logging.DEBUG if args.verbose else logging.WARNING)
    if args.list_devices:
        print(_sd().query_devices())
        return 0

    _load_env()
    api_key = os.environ.get("ELEVENLABS_API_KEY")
    agent_id = args.agent_id or os.environ.get("ELEVENLABS_AGENT_ID")
    if not api_key or not agent_id:
        print("❌ ELEVENLABS_API_KEY and ELEVENLABS_AGENT_ID must be set (repo .env).")
        return 2
    try:
        return run(agent_id, api_key, args.mode, args.input_device, args.output_device)
    except ImportError as exc:
        print(f"❌ missing dependency ({exc.name}); run: uv sync --inexact --extra elevenlabs")
        return 3


if __name__ == "__main__":
    sys.exit(main())
