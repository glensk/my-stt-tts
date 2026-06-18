"""Audio transport seam (R2-5): source mic audio + sink TTS audio over a *seam*.

The voice loop historically owned the mic + speaker directly via
:mod:`sounddevice`. That works for one local process but not for a whole-house
satellite or a remote browser. This module introduces an :class:`AudioTransport`
protocol so the loop can source mic frames and sink TTS PCM over **any** medium —
local sound card (:class:`LocalTransport`, the default) or a network link
(:class:`WebSocketTransport`).

The framing/handshake logic is **pure** (no network, no mic) so it is unit-tested
directly: :func:`encode_frame`/:func:`decode_frame` move float32 mono PCM as
little-endian int16 over a binary channel; :func:`make_handshake`/
:func:`check_handshake` gate a connection on an optional shared token. The actual
``websockets`` server is an optional extra (``my-stt-tts[transport]``) and is
imported lazily so the core package stays dependency-light.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import numpy as np

log = logging.getLogger("my_stt_tts.transport")

TRANSPORT_MODES = ("local", "websocket")

# Wire format. A binary frame carries raw PCM (int16 LE); a text frame carries a
# small JSON control envelope. The handshake is the first text frame from the
# client and gates the session on the optional shared token.
PROTOCOL_VERSION = 1
_INT16_SCALE = 32767.0


def encode_frame(pcm: np.ndarray) -> bytes:
    """Encode float32 mono PCM in [-1, 1] to little-endian int16 bytes (the wire format).

    Clipping guards against out-of-range samples; an empty array encodes to ``b""``.
    """
    arr = np.asarray(pcm, dtype=np.float32).ravel()
    if arr.size == 0:
        return b""
    clipped = np.clip(arr, -1.0, 1.0)
    return (clipped * _INT16_SCALE).astype("<i2").tobytes()


def decode_frame(data: bytes) -> np.ndarray:
    """Decode little-endian int16 bytes back to float32 mono PCM in [-1, 1].

    The inverse of :func:`encode_frame`. A trailing odd byte (a truncated sample)
    is dropped rather than raising, so a partial network read degrades gracefully.
    """
    if not data:
        return np.zeros(0, dtype=np.float32)
    if len(data) % 2:
        data = data[:-1]  # drop a truncated half-sample
    ints = np.frombuffer(data, dtype="<i2").astype(np.float32)
    return ints / _INT16_SCALE


def make_handshake(*, sample_rate: int, token: str | None = None, role: str = "satellite") -> str:
    """Build the client's opening control envelope (JSON text frame).

    ``role`` distinguishes a ``satellite`` (full-duplex mic+speaker) from a
    ``browser`` client; ``token`` (when set) must match the server's shared token.
    """
    env: dict[str, Any] = {
        "type": "hello",
        "version": PROTOCOL_VERSION,
        "sample_rate": int(sample_rate),
        "role": role,
    }
    if token:
        env["token"] = token
    return json.dumps(env)


def check_handshake(raw: str | bytes, *, token: str | None = None) -> dict[str, Any]:
    """Validate a client handshake; return the parsed envelope or raise ``ValueError``.

    Enforces the protocol version and, when the server configured a shared
    ``token``, that the client presented the same one. Never trusts unparsed
    input: malformed JSON, a wrong type, an unknown version, or a token mismatch
    all raise ``ValueError`` so the server can close the socket cleanly.
    """
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("handshake is not UTF-8 text") from exc
    try:
        env = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("handshake is not valid JSON") from exc
    if not isinstance(env, dict) or env.get("type") != "hello":
        raise ValueError("handshake must be a 'hello' envelope")
    if env.get("version") != PROTOCOL_VERSION:
        raise ValueError(f"unsupported protocol version {env.get('version')!r}")
    if token and env.get("token") != token:
        raise ValueError("handshake token mismatch")
    return env


def control_message(kind: str, **fields: Any) -> str:
    """Serialize a server→client control message (e.g. ``state``, ``bye``)."""
    return json.dumps({"type": kind, **fields})


@runtime_checkable
class AudioTransport(Protocol):
    """The seam the loop talks to instead of :mod:`sounddevice` directly.

    A transport is the mic *source* (``mic_frames`` yields captured PCM) and the
    TTS *sink* (``send_tts`` plays/forwards synthesized PCM). ``close`` releases
    any resources. Implementations: :class:`LocalTransport` (sound card) and
    :class:`WebSocketTransport` (network).
    """

    sample_rate: int

    def mic_frames(self) -> Iterator[np.ndarray]:
        """Yield float32 mono mic frames until the source ends."""
        ...

    def send_tts(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Sink one chunk of synthesized TTS PCM (play locally or forward)."""
        ...

    def close(self) -> None:
        """Release the underlying device / connection."""
        ...


class LocalTransport:
    """Default transport: the local sound card via :mod:`sounddevice` (today's behaviour).

    Mic capture and playback delegate to :mod:`my_stt_tts.audio`, so the existing
    local loop is unchanged — this just gives it the :class:`AudioTransport`
    surface so the network transports can stand in for it. ``mic_frames`` streams
    the live input device; ``send_tts`` plays a PCM chunk and blocks until done.
    """

    def __init__(self, sample_rate: int = 16000, *, frame_samples: int = 512) -> None:
        self.sample_rate = sample_rate
        self.frame_samples = frame_samples
        self._closed = False

    def mic_frames(self) -> Iterator[np.ndarray]:
        """Stream live mic frames from the default input device (blocking generator)."""
        import queue as _queue

        from . import audio

        sd = audio._sd()  # noqa: SLF001 — same package lazy accessor
        frames_q: _queue.Queue[np.ndarray] = _queue.Queue()

        def _callback(indata, _frames, _time, _status) -> None:  # noqa: ANN001
            frames_q.put(indata[:, 0].copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=_callback,
        ):
            while not self._closed:
                try:
                    yield frames_q.get(timeout=0.1)
                except _queue.Empty:
                    continue

    def send_tts(self, pcm: np.ndarray, sample_rate: int) -> None:
        """Play a chunk of TTS PCM on the local speaker, blocking until done."""
        from . import audio

        audio.play(np.asarray(pcm, dtype=np.float32), sample_rate)

    def close(self) -> None:
        self._closed = True
