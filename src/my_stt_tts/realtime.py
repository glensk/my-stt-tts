"""Speech-to-speech / realtime LLM option (R3-5): bypass the STT→LLM→TTS cascade.

The cascade (record → transcribe → stream an LLM → synthesize → play) has an
*irreducible* per-turn latency: each stage waits for the one before it. A realtime
**speech-to-speech** endpoint collapses that — you stream mic audio in and get
synthesized audio back over a single WebSocket, with the model doing ASR + reasoning
+ TTS internally and barging far earlier. This module adds that as an *option*,
behind config (``brain=realtime``), key-gated, with graceful fallback to the
cascade when no key/endpoint is configured.

It targets the **real OpenAI Realtime API** WS protocol:

* client→server: ``session.update`` (configure voice / audio formats / VAD),
  ``input_audio_buffer.append`` (base64 mic audio), ``input_audio_buffer.commit``,
  ``response.create`` (ask for a reply),
* server→client: ``response.audio.delta`` (base64 audio chunks to play),
  ``response.audio_transcript.delta`` (the spoken text, for the UI / history),
  ``response.done``, ``error``, plus the ``input_audio_buffer.speech_started`` /
  ``speech_stopped`` VAD events.

**Design for testability.** The protocol — building those client events and
decoding those server events, and the base64 ⇄ int16-PCM transcode — lives in
:class:`RealtimeProtocol` and is **pure** (no socket, no key). The actual WS
connect (:meth:`RealtimeClient.connect`) is isolated and lazy-imports
``websockets``. :func:`run_realtime_session` drives an
:class:`~my_stt_tts.transport.AudioTransport` against the protocol, so the tests
exercise the full mic→endpoint→audio-back round-trip against a **mocked realtime
WS server** — no real key, no network.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import threading
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np

from .events import bus
from .run_ledger import RunRecorder, emit_run, endpoint_seat, record_run, usage_fields

if TYPE_CHECKING:
    from .config import Config
    from .transport import AudioTransport

log = logging.getLogger("my_stt_tts.realtime")

# OpenAI Realtime PCM16 audio is 24 kHz mono little-endian int16.
REALTIME_SR = 24000
_INT16_SCALE = 32767.0


def pcm_to_base64(pcm: np.ndarray) -> str:
    """Encode float32 mono PCM in [-1, 1] as base64 little-endian int16 (the wire format)."""
    arr = np.asarray(pcm, dtype=np.float32).ravel()
    if arr.size == 0:
        return ""
    clipped = np.clip(arr, -1.0, 1.0)
    return base64.b64encode((clipped * _INT16_SCALE).astype("<i2").tobytes()).decode("ascii")


def base64_to_pcm(b64: str) -> np.ndarray:
    """Decode a base64 little-endian int16 audio chunk back to float32 mono PCM."""
    if not b64:
        return np.zeros(0, dtype=np.float32)
    try:
        raw = base64.b64decode(b64)
    except (ValueError, TypeError):
        return np.zeros(0, dtype=np.float32)
    if len(raw) % 2:
        raw = raw[:-1]
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / _INT16_SCALE


class RealtimeProtocol:
    """Build client events / decode server events for the OpenAI Realtime WS API.

    Pure string↔dict transforms (plus the base64 PCM transcode) — no socket, no
    key — so the whole protocol is unit-tested directly. Each builder returns the
    JSON **string** to send; :meth:`decode` parses one server frame into a small
    tagged dict the session loop reacts to.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-realtime-preview",
        voice: str = "alloy",
        audio_format: str = "pcm16",
        instructions: str = "",
    ) -> None:
        self.model = model
        self.voice = voice
        self.audio_format = audio_format
        self.instructions = instructions

    # --- client → server -------------------------------------------------

    def session_update(self) -> str:
        """The ``session.update`` event: configure voice, audio formats, and VAD.

        Sets server-side voice-activity detection (``turn_detection`` =
        ``server_vad``) so the endpoint segments turns itself, and pins the input
        and output audio formats. ``instructions`` carries the spoken-output system
        prompt so the realtime model speaks with the same persona as the cascade.
        """
        session: dict[str, Any] = {
            "modalities": ["audio", "text"],
            "voice": self.voice,
            "input_audio_format": self.audio_format,
            "output_audio_format": self.audio_format,
            "turn_detection": {"type": "server_vad"},
        }
        if self.instructions:
            session["instructions"] = self.instructions
        return json.dumps({"type": "session.update", "session": session})

    def append_audio(self, pcm: np.ndarray) -> str:
        """An ``input_audio_buffer.append`` event carrying base64 mic audio."""
        return json.dumps({"type": "input_audio_buffer.append", "audio": pcm_to_base64(pcm)})

    def commit_audio(self) -> str:
        """An ``input_audio_buffer.commit`` event (finalize the captured utterance)."""
        return json.dumps({"type": "input_audio_buffer.commit"})

    def create_response(self) -> str:
        """A ``response.create`` event asking the model to reply (audio + text)."""
        return json.dumps(
            {"type": "response.create", "response": {"modalities": ["audio", "text"]}}
        )

    def cancel_response(self) -> str:
        """A ``response.cancel`` event — stop the in-flight reply (barge-in)."""
        return json.dumps({"type": "response.cancel"})

    # --- server → client -------------------------------------------------

    def decode(self, raw: str | bytes) -> dict[str, Any]:
        """Parse one server frame into ``{"type": ..., ...}`` with decoded payloads.

        ``response.audio.delta`` carries ``pcm`` (float32 frames, decoded from the
        base64 chunk); ``response.audio_transcript.delta`` /``.done`` carry the
        spoken ``text``; ``error`` carries ``message``. Unknown / malformed frames
        return ``{"type": "unknown"}`` rather than raising, so a junk frame never
        kills the session.
        """
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"type": "unknown"}
        if not isinstance(msg, dict):
            return {"type": "unknown"}
        etype = str(msg.get("type", "unknown"))
        event: dict[str, Any] = {"type": etype}
        if etype == "response.audio.delta":
            event.update(
                pcm=base64_to_pcm(str(msg.get("delta", ""))),
                response_id=str(msg.get("response_id", "")),
            )
        elif etype in ("response.audio_transcript.delta", "response.text.delta"):
            event.update(
                text=str(msg.get("delta", "")), response_id=str(msg.get("response_id", ""))
            )
        elif etype == "response.audio_transcript.done":
            event.update(
                text=str(msg.get("transcript", "")),
                response_id=str(msg.get("response_id", "")),
            )
        elif etype in ("response.created", "response.done"):
            response = msg.get("response")
            response_id = response.get("id", "") if isinstance(response, dict) else ""
            event.update(response=response, response_id=str(response_id))
        elif etype == "error":
            err = msg.get("error", {})
            message = err.get("message") if isinstance(err, dict) else str(err)
            event["message"] = str(message or "realtime error")
        return event


class RealtimeError(RuntimeError):
    """Raised when the realtime endpoint is unavailable or returns an error."""


class RealtimeClient:
    """Thin WebSocket client for the OpenAI Realtime API (R3-5).

    The protocol (event encode/decode) is :class:`RealtimeProtocol`; this only
    adds the actual socket. :meth:`connect` is isolated + lazy (imports
    ``websockets`` only when called) so the core package never needs the dep and
    the tests can swap in a fake connection. ``available()`` reports whether a key
    is configured — :func:`make_realtime_brain` uses it to decide fallback.
    """

    def __init__(self, cfg: Config, *, recorder: RunRecorder = record_run) -> None:
        self.cfg = cfg
        self.recorder = recorder
        self.protocol = RealtimeProtocol(
            model=cfg.realtime_model,
            voice=cfg.realtime_voice,
            audio_format=cfg.realtime_audio_format,
            instructions=getattr(cfg, "system_prompt", ""),
        )

    def available(self) -> bool:
        """Whether a realtime API key + endpoint are configured (else fall back)."""
        return bool(self.cfg.realtime_api_key and self.cfg.realtime_url)

    async def connect(self) -> Any:
        """Open the realtime WebSocket (lazy ``websockets`` import). Returns the conn.

        The model is selected via the ``?model=`` query param and the key via the
        ``Authorization`` + ``OpenAI-Beta`` headers, per the OpenAI Realtime docs.
        Raises :class:`RealtimeError` without the ``transport`` extra or a key.
        """
        if not self.available():
            raise RealtimeError("realtime endpoint needs REALTIME_API_KEY / OPENAI_API_KEY")
        try:
            import websockets  # pylint: disable=import-outside-toplevel,import-error
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise RealtimeError(
                "realtime needs the 'transport' extra: uv sync --extra transport"
            ) from exc
        url = f"{self.cfg.realtime_url}?model={self.cfg.realtime_model}"
        headers = {
            "Authorization": f"Bearer {self.cfg.realtime_api_key}",
            "OpenAI-Beta": "realtime=v1",
        }
        return await websockets.connect(url, additional_headers=headers, max_size=None)


def run_realtime_session(
    transport: AudioTransport,
    cfg: Config,
    *,
    client: RealtimeClient | None = None,
    connection: Any | None = None,
    max_turns: int | None = None,
    recorder: RunRecorder = record_run,
) -> None:
    """Bridge a mic/audio :class:`AudioTransport` to a realtime endpoint (R3-5).

    Streams mic frames from ``transport`` into ``input_audio_buffer.append`` events
    and sinks the endpoint's ``response.audio.delta`` audio back through
    ``transport.send_tts`` — the STT→LLM→TTS cascade is entirely bypassed. Server
    VAD segments turns, so we simply forward mic audio and play whatever audio comes
    back, publishing the spoken transcript to the bus for the UI.

    The actual asyncio plumbing is in :func:`_run_realtime_async`; this wrapper runs
    it on its own event loop. ``connection`` (a duck-typed WS with ``send`` + async
    iteration) is injectable so tests drive a **mocked realtime server** — no key,
    no network. ``max_turns`` bounds the loop for tests (None = run until the mic
    source or the connection ends).
    """
    client = client or RealtimeClient(cfg, recorder=recorder)
    asyncio.run(_run_realtime_async(transport, cfg, client, connection, max_turns))


class _RealtimeLedger:
    """Track and asynchronously record OpenAI Realtime response attempts."""

    def __init__(self, cfg: Config, recorder: RunRecorder) -> None:
        self._cfg = cfg
        self._recorder = recorder
        self._attempts: dict[str, float] = {}
        self._tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _key(event: dict[str, Any]) -> str:
        return str(event.get("response_id") or "current")

    def observe(self, event: dict[str, Any]) -> None:
        """Start timing the response represented by its first server event."""
        etype = event.get("type")
        if isinstance(etype, str) and etype.startswith("response.") and etype != "response.done":
            key = self._key(event)
            if key == "current" and "current" not in self._attempts and len(self._attempts) == 1:
                return
            self._attempts.setdefault(key, time.perf_counter())

    def finish(self, event: dict[str, Any], *, error: str | None = None) -> None:
        """Record a completed/failed response without blocking the audio loop."""
        if error is not None and not self._attempts:
            return
        key = self._key(event)
        if key not in self._attempts:
            if "current" in self._attempts:
                key = "current"
            elif len(self._attempts) == 1:
                key = next(iter(self._attempts))
        started = self._attempts.pop(key, time.perf_counter())
        response = event.get("response")
        status = response.get("status") if isinstance(response, dict) else None
        failed = error or (str(status) if status not in (None, "completed") else None)
        row = self._base_row(started)
        row.update(usage_fields(response))
        row.setdefault("model", self._cfg.realtime_model)
        if failed:
            row.update(outcome=f"error:{failed}", ok=False, error=failed, error_message=failed)
        self._queue(row)

    def cancel_pending(self) -> None:
        """Record responses that ended without a terminal ``response.done`` event."""
        for key, started in list(self._attempts.items()):
            row = self._base_row(started)
            row.update(
                outcome="error:cancelled",
                ok=False,
                error="cancelled",
                error_message=f"realtime response {key} ended without response.done",
            )
            self._queue(row)
        self._attempts.clear()

    async def drain(self) -> None:
        """Wait for queued best-effort recorders when the socket session ends."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def _base_row(self, started: float) -> dict[str, object]:
        return {
            "provider": "openai",
            "seat": endpoint_seat(self._cfg.realtime_url),
            "purpose": "stt-brain",
            "outcome": "ok",
            "ok": True,
            "ms": int((time.perf_counter() - started) * 1000),
            "caller": "my_stt_tts.realtime",
            "requested_model": self._cfg.realtime_model,
            "model": self._cfg.realtime_model,
        }

    def _queue(self, row: dict[str, object]) -> None:
        task = asyncio.create_task(asyncio.to_thread(emit_run, self._recorder, row))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


def _start_mic_pump(
    transport: AudioTransport,
    conn: Any,
    proto: RealtimeProtocol,
    loop: asyncio.AbstractEventLoop,
    stop: threading.Event,
) -> None:
    """Start forwarding blocking microphone frames into the async socket."""

    def _pump() -> None:
        try:
            for frame in transport.mic_frames():
                if stop.is_set():
                    return
                arr = np.asarray(frame, dtype=np.float32).ravel()
                if arr.size:
                    future = asyncio.run_coroutine_threadsafe(
                        conn.send(proto.append_audio(arr)), loop
                    )
                    future.result(timeout=5.0)
        except Exception:  # pylint: disable=broad-exception-caught
            log.debug("realtime mic pump ended", exc_info=True)
        finally:
            stop.set()

    threading.Thread(target=_pump, daemon=True).start()


async def _run_realtime_async(
    transport: AudioTransport,
    cfg: Config,
    client: RealtimeClient,
    connection: Any | None,
    max_turns: int | None,
) -> None:
    """The async core of :func:`run_realtime_session` (see its docstring)."""
    conn = connection if connection is not None else await client.connect()
    proto = client.protocol
    await conn.send(proto.session_update())
    bus.state("listening", "realtime")

    # Pump mic frames into the endpoint in a background thread (the transport's
    # mic_frames() is a blocking generator); the main task reads server events.
    loop = asyncio.get_running_loop()
    stop = threading.Event()

    _start_mic_pump(transport, conn, proto, loop, stop)

    turns = 0
    ledger = _RealtimeLedger(cfg, client.recorder)

    try:
        async for raw in conn:
            event = proto.decode(raw)
            etype = event.get("type")
            ledger.observe(event)
            if etype == "response.audio.delta":
                pcm = event.get("pcm")
                if isinstance(pcm, np.ndarray) and pcm.size:
                    bus.state("speaking", "realtime")
                    transport.send_tts(pcm, REALTIME_SR)
            elif etype in ("response.audio_transcript.delta", "response.text.delta"):
                bus.response(str(event.get("text", "")), final=False)
            elif etype == "response.audio_transcript.done":
                bus.response(str(event.get("text", "")), final=True)
            elif etype == "input_audio_buffer.speech_started":
                bus.state("recording", "realtime")
            elif etype == "response.done":
                ledger.finish(event)
                bus.response("", final=True)
                bus.state("idle")
                turns += 1
                if max_turns is not None and turns >= max_turns:
                    break
            elif etype == "error":
                ledger.finish(event, error="realtime")
                log.error("realtime error: %s", event.get("message"))
                bus.log(str(event.get("message")), "error")
        # NB: a finished mic source (``stop``) must NOT end this loop — the model may
        # still be streaming its reply. We read server events until the connection
        # is exhausted or ``max_turns`` is hit; the mic pump stops independently.
    finally:
        ledger.cancel_pending()
        stop.set()
        with contextlib.suppress(Exception):
            await conn.close()
        await ledger.drain()


class RealtimeBrain:
    """A ``brain=realtime`` marker that drives the speech-to-speech loop (R3-5).

    The cascade's :class:`~my_stt_tts.brain.Brain` is a text streamer; a realtime
    brain works at the *audio* level, so it does not implement ``stream``. Instead
    it owns a :class:`RealtimeClient` and a :meth:`run` that bridges a transport to
    the endpoint (:func:`run_realtime_session`). ``__main__`` selects it when
    ``brain_mode == "realtime"`` and a key is configured, else it falls back to the
    cascade — see :func:`make_realtime_brain`.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = RealtimeClient(cfg)

    def available(self) -> bool:
        return self.client.available()

    def run(self, transport: AudioTransport, *, connection: Any | None = None) -> None:
        """Run the realtime speech-to-speech session over ``transport``."""
        run_realtime_session(transport, self.cfg, client=self.client, connection=connection)

    # A realtime brain has no text stream; keep the surface explicit for callers
    # that probe for cascade behaviour.
    def stream(self, user_text: str) -> Iterator[str]:
        del user_text
        raise RealtimeError("RealtimeBrain is audio-only; use run() over a transport")


def make_realtime_brain(cfg: Config) -> RealtimeBrain | None:
    """Build a :class:`RealtimeBrain` when realtime is selected AND keyed, else ``None``.

    The graceful-fallback gate (R3-5): returns a brain only when
    ``brain_mode == "realtime"`` *and* a key/endpoint is configured. ``None`` tells
    ``__main__`` to fall back to the STT→LLM→TTS cascade, with a clear log line, so
    a missing key degrades cleanly instead of erroring.
    """
    if getattr(cfg, "brain_mode", "cascade") != "realtime":
        return None
    brain = RealtimeBrain(cfg)
    if not brain.available():
        log.warning(
            "brain=realtime selected but no REALTIME_API_KEY/OPENAI_API_KEY; "
            "falling back to the STT->LLM->TTS cascade."
        )
        return None
    return brain
