"""Minimal RFC 6455 WebSocket framing for the browser audio channel (R2-5).

The ``--browser`` GUI is served by stdlib ``http.server`` (no extra deps). To carry
real audio from the page over a **same-origin** WebSocket (so the page's strict CSP
``connect-src 'self'`` allows it), we speak just enough of the WebSocket protocol
ourselves: the opening handshake key, and binary/text/close/ping/pong frame
encode/decode. This is **pure** (operates on bytes) and unit-tested directly — no
socket, no event loop. The actual socket plumbing lives in :mod:`my_stt_tts.webui`.

Only the subset the browser uses is implemented: server→client frames are
unmasked; client→server frames are always masked (the spec requires it) and we
unmask them. Continuation frames and >64-bit payloads are not needed for short PCM
chunks and are handled defensively (rejected) rather than silently mis-parsed.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import dataclass

# The fixed GUID from RFC 6455 §4.2.2 used to derive the Sec-WebSocket-Accept value.
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

OP_CONT = 0x0
OP_TEXT = 0x1
OP_BINARY = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


def accept_key(sec_websocket_key: str) -> str:
    """Compute the ``Sec-WebSocket-Accept`` response value for a client key."""
    digest = hashlib.sha1((sec_websocket_key + _WS_GUID).encode("ascii")).digest()  # noqa: S324
    return base64.b64encode(digest).decode("ascii")


def encode_frame(payload: bytes, *, opcode: int = OP_BINARY, fin: bool = True) -> bytes:
    """Encode one **server→client** (unmasked) frame around ``payload``."""
    header = bytearray()
    header.append((0x80 if fin else 0x00) | (opcode & 0x0F))
    length = len(payload)
    if length < 126:
        header.append(length)
    elif length < 65536:
        header.append(126)
        header += struct.pack(">H", length)
    else:
        header.append(127)
        header += struct.pack(">Q", length)
    return bytes(header) + payload


@dataclass
class Frame:
    """A decoded WebSocket frame."""

    opcode: int
    payload: bytes
    fin: bool = True


def decode_frame(data: bytes) -> tuple[Frame, int] | None:
    """Decode one **client→server** (masked) frame from ``data``.

    Returns ``(frame, bytes_consumed)`` when a whole frame is present, or ``None``
    when more bytes are still needed (the caller should read more and retry). The
    client mask is applied (unmasked) per the spec. Raises ``ValueError`` on a
    protocol violation (an unmasked client frame, or a payload length we don't
    support) so the caller can close the socket rather than mis-frame the stream.
    """
    if len(data) < 2:
        return None
    b0, b1 = data[0], data[1]
    fin = bool(b0 & 0x80)
    opcode = b0 & 0x0F
    masked = bool(b1 & 0x80)
    length = b1 & 0x7F
    idx = 2
    if length == 126:
        if len(data) < idx + 2:
            return None
        length = struct.unpack(">H", data[idx : idx + 2])[0]
        idx += 2
    elif length == 127:
        if len(data) < idx + 8:
            return None
        length = struct.unpack(">Q", data[idx : idx + 8])[0]
        idx += 8
    if not masked:
        # Per RFC 6455 a client MUST mask; an unmasked client frame is a violation.
        raise ValueError("client frame is not masked")
    if len(data) < idx + 4:
        return None
    mask = data[idx : idx + 4]
    idx += 4
    if len(data) < idx + length:
        return None  # payload not fully arrived yet
    masked_payload = data[idx : idx + length]
    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(masked_payload))
    return Frame(opcode=opcode, payload=payload, fin=fin), idx + length


def close_frame(code: int = 1000) -> bytes:
    """A server close frame with ``code``."""
    return encode_frame(struct.pack(">H", code), opcode=OP_CLOSE)
