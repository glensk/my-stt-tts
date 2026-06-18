"""WebSocket audio transport (R2-5): mic PCM in / TTS PCM out over the wire.

The framing/handshake/queue logic lives here and is **pure** (no real socket), so
it is unit-tested directly. The actual ``websockets`` server is an optional extra
(``my-stt-tts[transport]``) imported lazily by :func:`serve_websocket` so the core
package never grows the dependency.

A remote client (the :mod:`my_stt_tts.satellite` script or the browser GUI) opens
one connection, sends a JSON ``hello`` handshake, then streams binary mic frames
(int16 LE PCM, see :mod:`my_stt_tts.transport`). The server feeds those into the
existing pipeline via :class:`WebSocketTransport`, which is the mic *source*, and
forwards synthesized TTS PCM back to the client as binary frames (the audio sink).
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from .transport import (
    check_handshake,
    control_message,
    decode_frame,
    encode_frame,
)

log = logging.getLogger("my_stt_tts.ws_transport")

# A sentinel pushed onto the inbound queue to end ``mic_frames`` cleanly.
_EOF = object()


class WebSocketTransport:
    """An :class:`~my_stt_tts.transport.AudioTransport` backed by two queues.

    The network layer (the ``websockets`` connection handler) pushes decoded mic
    PCM via :meth:`feed_mic` and drains outbound TTS PCM via :meth:`iter_outbound`;
    the synchronous pipeline thread consumes :meth:`mic_frames` and produces via
    :meth:`send_tts`. Decoupling through queues lets the async socket and the
    blocking turn loop run in separate threads without either blocking the other.

    This object is deliberately transport-agnostic: the tests drive it with plain
    bytes, no socket required.
    """

    def __init__(self, sample_rate: int = 16000, *, max_queue: int = 512) -> None:
        self.sample_rate = sample_rate
        self._inbound: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._outbound: queue.Queue[bytes] = queue.Queue(maxsize=max_queue)
        self._closed = threading.Event()

    # --- network side -----------------------------------------------------

    def feed_mic(self, data: bytes) -> None:
        """Decode an inbound binary frame and queue it for the pipeline (network thread)."""
        if self._closed.is_set():
            return
        frame = decode_frame(data)
        if frame.size:
            try:
                self._inbound.put_nowait(frame)
            except queue.Full:
                log.warning("inbound mic queue full; dropping a frame")

    def iter_outbound(self, timeout: float = 0.1) -> bytes | None:
        """Pop the next outbound TTS frame (already encoded), or None on timeout."""
        try:
            return self._outbound.get(timeout=timeout)
        except queue.Empty:
            return None

    def end_mic(self) -> None:
        """Signal that no more mic frames will arrive (client closed)."""
        with _suppress_full():
            self._inbound.put_nowait(_EOF)

    # --- pipeline side (AudioTransport surface) ---------------------------

    def mic_frames(self) -> Iterator[np.ndarray]:
        """Yield mic frames fed by the network thread until EOF, then any close.

        Drains the queue first (so frames already buffered when the socket closes
        are still delivered); the ``_EOF`` sentinel — pushed by :meth:`end_mic`
        before :meth:`close` — is the clean terminator. ``_closed`` only ends the
        loop once the queue has run dry, so no buffered audio is dropped.
        """
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

    def send_tts(self, pcm: np.ndarray, sample_rate: int) -> None:  # noqa: ARG002
        """Encode a TTS PCM chunk and queue it to be sent over the socket."""
        data = encode_frame(np.asarray(pcm, dtype=np.float32))
        if not data:
            return
        with _suppress_full():
            self._outbound.put_nowait(data)

    def close(self) -> None:
        self._closed.set()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()


class _suppress_full:  # noqa: N801 — tiny context-manager helper
    """Swallow ``queue.Full`` (best-effort enqueue; a dropped frame is acceptable)."""

    def __enter__(self) -> _suppress_full:
        return self

    def __exit__(self, exc_type: type | None, *_exc: object) -> bool:
        return exc_type is not None and issubclass(exc_type, queue.Full)


class WsSession:
    """The per-connection handler logic, decoupled from the ``websockets`` server.

    ``handle(conn)`` is an ``async`` coroutine that: validates the handshake
    (optionally against ``token``), builds a :class:`WebSocketTransport`, runs
    ``on_session`` in a worker thread (the synchronous turn loop), then pumps
    inbound binary frames into the transport and outbound TTS frames back. It is
    written against a *duck-typed* connection (``recv``/``send``/``close`` + async
    iteration), so the tests drive it with a fake socket — no real network. The
    ``websockets`` server in :func:`serve_websocket` just hands it real sockets.
    """

    def __init__(
        self,
        on_session: Callable[[WebSocketTransport], None],
        *,
        token: str | None = None,
        sample_rate: int = 16000,
    ) -> None:
        self._on_session = on_session
        self._token = token
        self._sample_rate = sample_rate

    async def handle(self, conn: Any) -> WebSocketTransport | None:
        """Accept (or reject) ``conn`` and run the session; return its transport.

        Returns ``None`` when the handshake is rejected (the socket is closed with
        a policy-violation code), else the transport that bridged the session.
        """
        import asyncio

        transport = WebSocketTransport(sample_rate=self._sample_rate)
        try:
            hello = await conn.recv()
            check_handshake(hello, token=self._token)
        except Exception as exc:  # noqa: BLE001 - reject any bad/un-authed client
            log.warning("rejecting client: %s", exc)
            await conn.close(code=1008, reason="handshake rejected")
            return None
        await conn.send(control_message("ready", sample_rate=self._sample_rate))
        loop = asyncio.get_running_loop()
        worker = threading.Thread(target=self._on_session, args=(transport,), daemon=True)
        worker.start()

        async def _pump_out() -> None:
            while not transport.closed:
                data = await loop.run_in_executor(None, transport.iter_outbound, 0.1)
                if data:
                    await conn.send(data)

        out_task = asyncio.ensure_future(_pump_out())
        try:
            async for message in conn:
                if isinstance(message, bytes):
                    transport.feed_mic(message)
                # text frames are client control messages; ignored for now
        finally:
            transport.end_mic()
            transport.close()
            out_task.cancel()
        return transport


def serve_websocket(
    on_session: Callable[[WebSocketTransport], None],
    *,
    host: str = "0.0.0.0",  # noqa: S104 — satellites connect from the LAN by design
    port: int = 8770,
    token: str | None = None,
    sample_rate: int = 16000,
) -> None:
    """Run a blocking ``websockets`` server; bridge each client via :class:`WsSession`.

    Requires the optional ``transport`` extra; raises a clear error if it is missing.
    """
    try:
        import asyncio

        import websockets
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "WebSocket transport needs the 'transport' extra: uv sync --extra transport"
        ) from exc

    session = WsSession(on_session, token=token, sample_rate=sample_rate)

    async def _on_conn(conn: Any) -> None:  # websockets wants a handler returning None
        await session.handle(conn)

    async def _main() -> None:
        async with websockets.serve(_on_conn, host, port, max_size=None):
            log.info("WebSocket audio transport listening on ws://%s:%d", host, port)
            await asyncio.Future()  # run forever

    asyncio.run(_main())
