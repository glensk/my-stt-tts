"""Telephony reach (R3-9): answer a phone call via Twilio Media Streams.

Twilio's `<Stream>` TwiML opens a **WebSocket** to your server and pushes the
caller's audio as a sequence of JSON events carrying **base64-encoded μ-law
(G.711) 8 kHz mono** media frames; you push synthesized speech back the same way.
This module bridges that wire to the existing pipeline so the *same* STT → LLM →
TTS loop (or the realtime brain) can take a phone call — no new pipeline.

Two layers, both **pure** (no Twilio, no socket) so they unit-test directly:

* :func:`ulaw_encode` / :func:`ulaw_decode` — the ITU-T G.711 μ-law companding
  algorithm between int16 linear PCM and 8-bit μ-law, plus :func:`resample_8k_16k`
  / :func:`resample_16k_8k` so Twilio's 8 kHz meets the pipeline's 16 kHz.
* :class:`TwilioMediaStreamSerializer` — encodes/decodes the Twilio WS event
  protocol (``connected`` / ``start`` / ``media`` / ``stop`` inbound; ``media``
  outbound, keyed by the call's ``streamSid``), turning inbound media into float32
  mono PCM frames (resampled to the loop rate) and outbound PCM into Twilio media
  envelopes.

:class:`TwilioTransport` is the :class:`~my_stt_tts.transport.AudioTransport` the
serializer drives — queue-backed, exactly like the WebSocket/WebRTC transports —
and :func:`serve_twilio` runs the actual ``websockets`` server (the optional
``transport`` extra), bridging each call into
:func:`~my_stt_tts.net_loop.run_transport_session`.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.telephony")

# Twilio Media Streams are fixed at 8 kHz mono μ-law (G.711). The pipeline runs at
# 16 kHz, so we resample at the boundary.
TWILIO_SR = 8000
_BIAS = 0x84  # μ-law companding bias (132)
_CLIP = 8159  # μ-law clip level in the 13-bit (sample >> 2) magnitude domain


# --- G.711 μ-law companding (pure numpy, ITU-T G.711) --------------------------


def ulaw_decode(data: bytes) -> np.ndarray:
    """Decode 8-bit μ-law bytes back to int16 linear PCM (ITU-T G.711).

    The standard G.711 μ-law expansion, byte-for-byte identical to
    ``audioop.ulaw2lin`` (width 2) across all 256 codes — and, unlike ``audioop``,
    it survives Python 3.13 (which dropped that module). The BIAS is in the
    16-bit-scaled domain so this yields full int16-range PCM directly (e.g. byte
    ``0x00`` -> ``-32124``). The inverse of :func:`ulaw_encode`. Empty in -> empty
    out; never raises on a partial read.
    """
    if not data:
        return np.zeros(0, dtype=np.int16)
    ulaw = np.frombuffer(data, dtype=np.uint8).astype(np.int32)
    ulaw = ~ulaw & 0xFF
    sign = ulaw & 0x80
    exponent = (ulaw >> 4) & 0x07
    mantissa = ulaw & 0x0F
    magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
    linear = np.where(sign != 0, -magnitude, magnitude)
    return np.clip(linear, -32768, 32767).astype(np.int16)


def _build_encode_lut() -> np.ndarray:
    """Precompute the full int16 → μ-law byte table, as the exact inverse of decode.

    Each μ-law byte expands (via :func:`ulaw_decode`) to a known linear level; the
    encoder is therefore *defined* as "pick the byte whose decoded value is nearest
    to the sample". That makes ``decode(encode(x))`` the closest representable
    level to ``x`` (a true nearest quantizer) and guarantees Twilio — which uses
    the same standard G.711 expansion — reconstructs exactly what we intended.
    Built once at import time; applied at call time by a single numpy gather.
    """
    levels = ulaw_decode(bytes(range(256))).astype(np.int32)  # the 256 decode levels
    # Index the table by the sample's uint16 bit-pattern so ``_ENCODE_LUT[x & 0xFFFF]``
    # gathers correctly for negative samples too: entry k is the byte for int16 k.
    samples = np.arange(65536, dtype=np.uint16).astype(np.int16).astype(np.int32)
    nearest = np.abs(samples[:, None] - levels[None, :]).argmin(axis=1)
    return nearest.astype(np.uint8)


_ENCODE_LUT = _build_encode_lut()


def ulaw_encode(pcm16: np.ndarray) -> bytes:
    """Encode int16 linear PCM to 8-bit μ-law bytes (ITU-T G.711).

    A single numpy gather through the precomputed :data:`_ENCODE_LUT` (the exact
    inverse of :func:`ulaw_decode`), so it is fast *and* a true nearest quantizer:
    ``ulaw_decode(ulaw_encode(x))`` is the closest representable μ-law level to x.
    Empty in -> empty out.
    """
    samples = np.asarray(pcm16, dtype=np.int16).ravel()
    if samples.size == 0:
        return b""
    return _ENCODE_LUT[samples.astype(np.uint16)].tobytes()


def _resample(arr: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Linear-resample float32 mono PCM (dependency-free). Identity when rates match."""
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if from_sr == to_sr or arr.size == 0:
        return arr
    n_out = max(1, int(round(arr.size * to_sr / from_sr)))
    x_new = np.linspace(0.0, arr.size - 1, n_out)
    return np.interp(x_new, np.arange(arr.size), arr).astype(np.float32)


def resample_8k_16k(arr: np.ndarray) -> np.ndarray:
    """Upsample 8 kHz (Twilio) float32 PCM to 16 kHz (the pipeline rate)."""
    return _resample(arr, TWILIO_SR, 16000)


def resample_16k_8k(arr: np.ndarray) -> np.ndarray:
    """Downsample 16 kHz (pipeline) float32 PCM to 8 kHz (Twilio)."""
    return _resample(arr, 16000, TWILIO_SR)


_INT16_SCALE = 32767.0


def pcm_float_to_int16(pcm: np.ndarray) -> np.ndarray:
    """Clip + scale float32 mono PCM in [-1, 1] to int16."""
    arr = np.clip(np.asarray(pcm, dtype=np.float32).ravel(), -1.0, 1.0)
    return (arr * _INT16_SCALE).astype(np.int16)


def int16_to_pcm_float(samples: np.ndarray) -> np.ndarray:
    """Scale int16 PCM back to float32 mono in [-1, 1]."""
    return np.asarray(samples, dtype=np.float32) / _INT16_SCALE


# --- Twilio Media Streams event protocol ---------------------------------------


class TwilioMediaStreamSerializer:
    """Encode/decode the Twilio Media Streams WS event protocol (R3-9).

    Stateful so it can latch the ``streamSid`` from the ``start`` event (Twilio
    requires it on every outbound ``media`` frame) and track readiness. All
    methods are pure string/bytes/numpy transforms — no socket — so the protocol
    is unit-tested directly with fakes.

    Inbound (Twilio → us): :meth:`decode` parses one JSON text frame into a small
    tagged dict; for a ``media`` event the base64 μ-law payload is decoded to
    **float32 mono PCM resampled to ``sample_rate``** ready for the pipeline.

    Outbound (us → Twilio): :meth:`encode_media` turns one chunk of float32
    pipeline PCM into a Twilio ``media`` JSON envelope (downsampled to 8 kHz,
    μ-law-encoded, base64-wrapped, keyed by the latched ``streamSid``).
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self.started = False
        self.stopped = False

    def decode(self, raw: str | bytes) -> dict[str, Any]:
        """Parse one inbound Twilio WS frame into ``{"event": ..., ...}``.

        Recognised events: ``connected``, ``start`` (latches ``streamSid`` /
        ``callSid``), ``media`` (decodes the payload to ``pcm`` float32 frames),
        ``stop`` (marks the call ended), ``mark``. An unknown / malformed frame
        returns ``{"event": "unknown"}`` rather than raising, so a junk frame
        never kills the call.
        """
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return {"event": "unknown"}
        if not isinstance(msg, dict):
            return {"event": "unknown"}
        event = str(msg.get("event", "unknown"))
        if event == "start":
            start = msg.get("start", {})
            self.stream_sid = msg.get("streamSid") or start.get("streamSid")
            self.call_sid = start.get("callSid")
            self.started = True
            return {"event": "start", "stream_sid": self.stream_sid, "call_sid": self.call_sid}
        if event == "media":
            payload = (msg.get("media") or {}).get("payload", "")
            pcm = self.decode_media_payload(payload)
            return {"event": "media", "pcm": pcm}
        if event == "stop":
            self.stopped = True
            return {"event": "stop"}
        return {"event": event}

    def decode_media_payload(self, payload: str) -> np.ndarray:
        """Decode a base64 μ-law payload to float32 mono PCM at ``sample_rate``."""
        if not payload:
            return np.zeros(0, dtype=np.float32)
        try:
            ulaw = base64.b64decode(payload)
        except (ValueError, TypeError):
            return np.zeros(0, dtype=np.float32)
        linear = int16_to_pcm_float(ulaw_decode(ulaw))
        return _resample(linear, TWILIO_SR, self.sample_rate)

    def encode_media(self, pcm: np.ndarray) -> str | None:
        """Encode float32 pipeline PCM into a Twilio outbound ``media`` JSON frame.

        Downsamples to 8 kHz, μ-law-encodes, base64-wraps, and keys the envelope
        with the latched ``streamSid``. Returns ``None`` for blank audio or before
        the ``start`` event has latched a ``streamSid`` (nothing to send yet).
        """
        if self.stream_sid is None:
            return None
        arr = np.asarray(pcm, dtype=np.float32).ravel()
        if arr.size == 0:
            return None
        ulaw = ulaw_encode(pcm_float_to_int16(resample_16k_8k(arr)))
        if not ulaw:
            return None
        payload = base64.b64encode(ulaw).decode("ascii")
        return json.dumps(
            {"event": "media", "streamSid": self.stream_sid, "media": {"payload": payload}}
        )

    def encode_mark(self, name: str) -> str | None:
        """A Twilio ``mark`` frame (used to detect when our audio finished playing)."""
        if self.stream_sid is None:
            return None
        return json.dumps({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": name}})

    def encode_clear(self) -> str | None:
        """A Twilio ``clear`` frame — flush buffered outbound audio (barge-in)."""
        if self.stream_sid is None:
            return None
        return json.dumps({"event": "clear", "streamSid": self.stream_sid})


# --- the AudioTransport the serializer drives ----------------------------------

_EOF = object()


class TwilioTransport:
    """An :class:`~my_stt_tts.transport.AudioTransport` for a Twilio call (R3-9).

    Queue-backed, like the WebSocket/WebRTC transports: the network side pushes
    decoded mic PCM via :meth:`feed_mic` and drains outbound media via
    :meth:`iter_outbound`; the synchronous pipeline thread consumes
    :meth:`mic_frames` and produces via :meth:`send_tts`. Decoupling through queues
    lets the async socket and the blocking turn loop run without blocking each
    other. The serializer does the μ-law / base64 / resample transcode; this just
    moves frames. Pure (no socket) so tests drive it directly.
    """

    def __init__(self, serializer: TwilioMediaStreamSerializer, *, max_queue: int = 1024) -> None:
        self.serializer = serializer
        self.sample_rate = serializer.sample_rate
        self._inbound: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._outbound: queue.Queue[str] = queue.Queue(maxsize=max_queue)
        self._closed = threading.Event()

    # --- network side -----------------------------------------------------

    def feed_mic(self, pcm: np.ndarray) -> None:
        """Queue already-decoded mic PCM frames for the pipeline (network thread)."""
        if self._closed.is_set():
            return
        frame = np.asarray(pcm, dtype=np.float32).ravel()
        if frame.size:
            try:
                self._inbound.put_nowait(frame)
            except queue.Full:
                log.warning("inbound Twilio mic queue full; dropping a frame")

    def iter_outbound(self, timeout: float = 0.1) -> str | None:
        """Pop the next outbound Twilio media JSON frame, or None on timeout."""
        try:
            return self._outbound.get(timeout=timeout)
        except queue.Empty:
            return None

    def end_mic(self) -> None:
        """Signal that no more mic frames will arrive (the caller hung up)."""
        with _suppress_full():
            self._inbound.put_nowait(_EOF)

    # --- pipeline side (AudioTransport surface) ---------------------------

    def mic_frames(self) -> Iterator[np.ndarray]:
        """Yield mic frames fed by the network thread until EOF / close."""
        while True:
            try:
                item = self._inbound.get(timeout=0.1)
            except queue.Empty:
                if self._closed.is_set():
                    return
                continue
            if item is _EOF:
                return
            yield item

    def send_tts(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Transcode a TTS PCM chunk to a Twilio media frame and queue it outbound."""
        arr = np.asarray(pcm, dtype=np.float32).ravel()
        if sample_rate != self.sample_rate and arr.size:
            arr = _resample(arr, sample_rate, self.sample_rate)
        frame = self.serializer.encode_media(arr)
        if frame is None:
            return
        with _suppress_full():
            self._outbound.put_nowait(frame)

    def close(self) -> None:
        self._closed.set()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()


class _suppress_full:  # noqa: N801 — tiny queue.Full swallow
    def __enter__(self) -> _suppress_full:
        return self

    def __exit__(self, exc_type: type | None, *_exc: object) -> bool:
        return exc_type is not None and issubclass(exc_type, queue.Full)


class TwilioSession:
    """Per-call handler, decoupled from the ``websockets`` server (R3-9).

    ``handle(conn)`` is an ``async`` coroutine that: builds a serializer +
    :class:`TwilioTransport`, runs ``on_session`` (the synchronous turn loop) in a
    worker thread, then pumps inbound Twilio JSON frames into the transport
    (decoding media to PCM) and outbound media frames back. Written against a
    *duck-typed* connection (``send`` + async iteration), so tests drive it with a
    fake socket — no real Twilio, no network.
    """

    def __init__(
        self,
        on_session: Callable[[TwilioTransport], None],
        *,
        sample_rate: int = 16000,
    ) -> None:
        self._on_session = on_session
        self._sample_rate = sample_rate

    async def handle(self, conn: Any) -> TwilioTransport:
        """Bridge one Twilio Media Stream call to the pipeline; return its transport."""
        import asyncio

        serializer = TwilioMediaStreamSerializer(sample_rate=self._sample_rate)
        transport = TwilioTransport(serializer)
        loop = asyncio.get_running_loop()
        worker = threading.Thread(target=self._on_session, args=(transport,), daemon=True)
        worker.start()

        async def _pump_out() -> None:
            while not transport.closed:
                frame = await loop.run_in_executor(None, transport.iter_outbound, 0.1)
                if frame is not None:
                    await conn.send(frame)

        out_task = asyncio.ensure_future(_pump_out())
        try:
            async for message in conn:
                event = serializer.decode(message)
                kind = event.get("event")
                if kind == "media":
                    transport.feed_mic(event["pcm"])
                elif kind == "stop":
                    break
        finally:
            transport.end_mic()
            transport.close()
            out_task.cancel()
        return transport


def serve_twilio(
    on_session: Callable[[TwilioTransport], None],
    *,
    host: str = "0.0.0.0",  # noqa: S104 — Twilio connects from the public internet
    port: int = 8771,
    sample_rate: int = 16000,
) -> None:
    """Run a blocking ``websockets`` server bridging Twilio calls to the pipeline.

    Point a Twilio ``<Stream url="wss://YOUR_HOST:PORT/">`` at this server. Needs
    the optional ``transport`` extra (``websockets``); raises a clear error if it
    is missing.
    """
    try:
        import asyncio

        import websockets
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Twilio telephony needs the 'transport' extra: uv sync --extra transport"
        ) from exc

    session = TwilioSession(on_session, sample_rate=sample_rate)

    async def _on_conn(conn: Any) -> None:
        await session.handle(conn)

    async def _main() -> None:
        async with websockets.serve(_on_conn, host, port, max_size=None):
            log.info("Twilio Media Streams server listening on ws://%s:%d", host, port)
            await asyncio.Future()  # run forever

    asyncio.run(_main())
