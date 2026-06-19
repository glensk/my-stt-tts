"""True WebRTC audio transport (R3-1): Opus + jitter buffer + NAT traversal.

The R2-5 network path ships **raw int16 PCM over a hand-rolled WebSocket** — no
packet-loss concealment, no jitter buffering, no NAT traversal. For a real remote
satellite / browser over the open internet that is fragile. This module adds a
third :class:`~my_stt_tts.transport.AudioTransport` backed by **aiortc** (the
optional ``webrtc`` extra): a real ``RTCPeerConnection`` negotiating **Opus**, with
aiortc providing the jitter buffer + RTP/SRTP + ICE (STUN/TURN) NAT traversal for
free. The browser path uses a real ``RTCPeerConnection`` +
``getUserMedia({audio:{echoCancellation:true}})`` instead of raw PCM; the WS PCM
path stays as a fallback.

**Design for testability.** aiortc is async and pulls Opus media tracks; the
synchronous pipeline (:func:`~my_stt_tts.net_loop.run_transport_session`) runs in a
thread. The *bridge* between the two — a queue-backed object that is the mic
*source* (``mic_frames``) and TTS *sink* (``send_tts``) — is **pure** (plain numpy
+ queues, no aiortc) and unit-tested directly (:class:`WebRtcTransport`). The
*signaling* (SDP offer→answer) is factored into :func:`negotiate_answer`, written
against a duck-typed peer-connection so the tests drive it with a fake (no real
ICE, no STUN, no sockets). The aiortc-specific media plumbing
(:class:`_PcmTrack`, :func:`run_webrtc_offer`) is isolated and lazy-imported so the
core package never grows the dependency.
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Any

import numpy as np

log = logging.getLogger("my_stt_tts.webrtc_transport")

# WebRTC media is canonically 48 kHz; aiortc decodes/encodes Opus at this rate.
WEBRTC_SR = 48000
_EOF = object()


def _resample(arr: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    """Linear-resample float32 mono PCM (dependency-free). Identity when rates match."""
    arr = np.asarray(arr, dtype=np.float32).ravel()
    if from_sr == to_sr or arr.size == 0:
        return arr
    n_out = max(1, int(round(arr.size * to_sr / from_sr)))
    x_new = np.linspace(0.0, arr.size - 1, n_out)
    return np.interp(x_new, np.arange(arr.size), arr).astype(np.float32)


class WebRtcTransport:
    """An :class:`~my_stt_tts.transport.AudioTransport` bridging WebRTC ⇄ the pipeline.

    The aiortc media side pushes **decoded** mic PCM (already Opus-decoded + jitter-
    buffered by aiortc) via :meth:`feed_mic` and drains outbound TTS PCM via
    :meth:`next_tts`; the synchronous pipeline thread consumes :meth:`mic_frames`
    and produces via :meth:`send_tts`. Both sides hand off through bounded queues so
    the async peer connection and the blocking turn loop never block each other —
    the same decoupling as :class:`~my_stt_tts.ws_transport.WebSocketTransport`, but
    the wire is real WebRTC.

    PCM is resampled between the WebRTC 48 kHz media rate and the pipeline
    ``sample_rate`` at the boundary, so the loop sees mono frames at its own rate.
    This object is deliberately aiortc-free: tests drive it with plain numpy.
    """

    def __init__(self, sample_rate: int = 16000, *, max_queue: int = 512) -> None:
        self.sample_rate = sample_rate
        self._inbound: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._outbound: queue.Queue[Any] = queue.Queue(maxsize=max_queue)
        self._closed = threading.Event()

    # --- WebRTC media side -------------------------------------------------

    def feed_mic(self, pcm: np.ndarray, sample_rate: int = WEBRTC_SR) -> None:
        """Queue Opus-decoded mic PCM for the pipeline (resampled to the loop rate)."""
        if self._closed.is_set():
            return
        frame = _resample(np.asarray(pcm, dtype=np.float32).ravel(), sample_rate, self.sample_rate)
        if frame.size:
            try:
                self._inbound.put_nowait(frame)
            except queue.Full:
                log.warning("inbound WebRTC mic queue full; dropping a frame")

    def next_tts(self, timeout: float = 0.1) -> tuple[np.ndarray, int] | None:
        """Pop the next outbound TTS chunk ``(pcm, sample_rate)`` for the media track."""
        try:
            item = self._outbound.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _EOF:
            return None
        return item  # type: ignore[no-any-return]

    def end_mic(self) -> None:
        """Signal that no more mic frames will arrive (peer closed)."""
        with _suppress_full():
            self._inbound.put_nowait(_EOF)

    # --- pipeline side (AudioTransport surface) ----------------------------

    def mic_frames(self) -> Any:
        """Yield mic frames fed by the media side until EOF / close (generator)."""
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
        """Queue a TTS PCM chunk (resampled to 48 kHz) for the outbound media track."""
        arr = _resample(np.asarray(pcm, dtype=np.float32).ravel(), sample_rate, WEBRTC_SR)
        if not arr.size:
            return
        with _suppress_full():
            self._outbound.put_nowait((arr, WEBRTC_SR))

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


async def negotiate_answer(pc: Any, offer: dict[str, str]) -> dict[str, str]:
    """Apply a remote SDP ``offer`` to ``pc`` and return the local SDP answer (R3-1).

    The browser-initiated WebRTC handshake: set the remote description from the
    offer, create + set the local answer, and return it as a plain ``{"sdp", "type"}``
    dict to send back over the signaling channel. Written against a duck-typed
    peer-connection (``setRemoteDescription`` / ``createAnswer`` /
    ``setLocalDescription`` + a ``localDescription`` with ``.sdp`` / ``.type``), so
    the tests drive it with a fake — no real ICE/STUN/DTLS. The aiortc
    ``RTCSessionDescription`` is lazy-imported only when a real ``pc`` needs it.
    """
    desc = _session_description(offer["sdp"], offer.get("type", "offer"))
    await pc.setRemoteDescription(desc)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    local = pc.localDescription
    return {"sdp": local.sdp, "type": local.type}


def _session_description(sdp: str, sdp_type: str) -> Any:
    """Build an ``RTCSessionDescription`` (aiortc) or a tiny stand-in for tests."""
    try:
        from aiortc import RTCSessionDescription

        return RTCSessionDescription(sdp=sdp, type=sdp_type)
    except ImportError:  # no webrtc extra -> a duck-typed stand-in (used in tests)

        class _Desc:  # noqa: N801
            def __init__(self, s: str, t: str) -> None:
                self.sdp = s
                self.type = t

        return _Desc(sdp, sdp_type)


def webrtc_available() -> bool:
    """Whether the ``webrtc`` extra (aiortc) is importable on this machine."""
    try:
        import importlib.util

        return importlib.util.find_spec("aiortc") is not None
    except Exception:
        return False


async def run_webrtc_offer(
    transport: WebRtcTransport,
    offer: dict[str, str],
    *,
    ice_servers: list[str] | None = None,
) -> dict[str, str]:
    """Build a real aiortc ``RTCPeerConnection`` for ``offer`` and return the answer.

    Wires the inbound Opus track to :meth:`WebRtcTransport.feed_mic` and attaches an
    outbound :class:`_PcmTrack` that pulls TTS PCM from the transport — so the
    browser's mic flows in and the synthesized reply flows back, both as Opus over
    a real peer connection with aiortc's jitter buffer + ICE NAT traversal. Needs
    the ``webrtc`` extra; raises a clear error if it is missing.
    """
    try:
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection
        from aiortc.mediastreams import MediaStreamError
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "WebRTC transport needs the 'webrtc' extra: uv sync --extra webrtc"
        ) from exc

    servers = [RTCIceServer(urls=u) for u in (ice_servers or ["stun:stun.l.google.com:19302"])]
    pc = RTCPeerConnection(RTCConfiguration(iceServers=servers))
    pc.addTrack(_make_pcm_track(transport))

    @pc.on("track")
    def _on_track(track: Any) -> None:  # noqa: ANN202
        if track.kind != "audio":
            return

        async def _pump() -> None:
            try:
                while not transport.closed:
                    frame = await track.recv()
                    transport.feed_mic(_frame_to_mono(frame), int(frame.sample_rate))
            except MediaStreamError:
                transport.end_mic()

        import asyncio

        asyncio.ensure_future(_pump())

    return await negotiate_answer(pc, offer)


def _frame_to_mono(frame: Any) -> np.ndarray:
    """Convert an aiortc ``AudioFrame`` to float32 mono PCM in [-1, 1]."""
    arr = frame.to_ndarray()
    arr = arr.astype(np.float32) / 32768.0 if arr.dtype == np.int16 else arr.astype(np.float32)
    if arr.ndim > 1:  # (channels, samples) or interleaved -> average to mono
        arr = arr.mean(axis=0) if arr.shape[0] < arr.shape[1] else arr.mean(axis=1)
    return arr.ravel()


def _make_pcm_track(transport: WebRtcTransport) -> Any:  # pragma: no cover - needs aiortc
    """Build an aiortc audio track that emits the transport's outbound TTS as Opus.

    aiortc calls ``recv`` to pull the next 20 ms ``AudioFrame``; the track drains the
    transport's outbound TTS queue (padding with silence when idle) and returns a
    48 kHz mono frame, which aiortc encodes to Opus. Defined as a factory so the
    module imports without aiortc; the real ``MediaStreamTrack`` subclass is created
    only when the ``webrtc`` extra is present.
    """
    import fractions

    from aiortc.mediastreams import MediaStreamTrack
    from av import AudioFrame

    class _PcmTrack(MediaStreamTrack):  # type: ignore[misc, valid-type]
        kind = "audio"

        def __init__(self, transport: WebRtcTransport) -> None:
            super().__init__()
            self._transport = transport
            self._buf = np.zeros(0, dtype=np.float32)
            self._pts = 0
            self._samples = 960  # 20 ms @ 48 kHz

        async def recv(self) -> Any:
            while self._buf.size < self._samples and not self._transport.closed:
                item = self._transport.next_tts(timeout=0.02)
                if item is None:
                    break
                pcm, _sr = item
                self._buf = np.concatenate([self._buf, pcm])
            if self._buf.size < self._samples:  # pad with silence to keep timing smooth
                self._buf = np.concatenate(
                    [self._buf, np.zeros(self._samples - self._buf.size, dtype=np.float32)]
                )
            chunk = self._buf[: self._samples]
            self._buf = self._buf[self._samples :]
            pcm16 = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).reshape(1, -1)
            frame = AudioFrame.from_ndarray(pcm16, format="s16", layout="mono")
            frame.sample_rate = WEBRTC_SR
            frame.pts = self._pts
            frame.time_base = fractions.Fraction(1, WEBRTC_SR)
            self._pts += self._samples
            return frame

    return _PcmTrack(transport)
