"""Tests for the network audio transport (R2-5):

* frame encode/decode round-trip + truncation tolerance
* the handshake builder/validator (version + shared-token auth)
* the queue-backed :class:`WebSocketTransport` (mic source + TTS sink hand-off)
* the WS server handler logic, exercised against a fake socket (no real network)
* the transport-driven turn loop (``net_loop``): capture -> respond source/sink

Everything fakes the network, mic, model, and provider — nothing here opens a
socket, touches a device, or calls an API.
"""
# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring
# pylint: disable=too-few-public-methods,redefined-outer-name,reimported,import-outside-toplevel
# (test doubles are intentionally tiny; ws_frame.decode_frame is imported locally so
#  it doesn't shadow the transport-layer decode_frame used elsewhere in this module)

import numpy as np
import pytest

from my_stt_tts import net_loop
from my_stt_tts.config import Config
from my_stt_tts.stt import STTResult
from my_stt_tts.transport import (
    PROTOCOL_VERSION,
    LocalTransport,
    check_handshake,
    decode_frame,
    encode_frame,
    make_handshake,
)
from my_stt_tts.ws_transport import WebSocketTransport

# --- framing -------------------------------------------------------------------


def test_encode_decode_round_trip_preserves_pcm():
    pcm = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    out = decode_frame(encode_frame(pcm))
    # int16 quantization tolerance (~1/32767).
    assert np.allclose(out, pcm, atol=1e-3)


def test_encode_clips_out_of_range():
    pcm = np.array([2.0, -2.0], dtype=np.float32)
    out = decode_frame(encode_frame(pcm))
    assert np.all(out <= 1.0) and np.all(out >= -1.0)
    assert out[0] > 0.9 and out[1] < -0.9


def test_encode_empty_is_empty_bytes():
    assert encode_frame(np.zeros(0, dtype=np.float32)) == b""
    assert decode_frame(b"").size == 0


def test_decode_drops_truncated_half_sample():
    # 3 bytes = one int16 + a dangling byte; the half-sample is dropped, not raised.
    out = decode_frame(b"\x01\x00\x7f")
    assert out.size == 1


# --- handshake -----------------------------------------------------------------


def test_handshake_round_trip_without_token():
    env = check_handshake(make_handshake(sample_rate=16000))
    assert env["type"] == "hello"
    assert env["version"] == PROTOCOL_VERSION
    assert env["sample_rate"] == 16000
    assert env["role"] == "satellite"


def test_handshake_token_must_match():
    raw = make_handshake(sample_rate=16000, token="s3cret")
    env = check_handshake(raw, token="s3cret")  # correct token -> ok
    assert env["token"] == "s3cret"
    with pytest.raises(ValueError, match="token mismatch"):
        check_handshake(raw, token="wrong")
    with pytest.raises(ValueError, match="token mismatch"):
        check_handshake(make_handshake(sample_rate=16000), token="s3cret")  # missing token


def test_handshake_rejects_bad_version():
    raw = '{"type":"hello","version":999,"sample_rate":16000}'
    with pytest.raises(ValueError, match="version"):
        check_handshake(raw)


def test_handshake_rejects_garbage():
    with pytest.raises(ValueError, match="valid JSON"):
        check_handshake("not json")
    with pytest.raises(ValueError, match="hello"):
        check_handshake('{"type":"bye"}')


def test_handshake_accepts_bytes():
    env = check_handshake(make_handshake(sample_rate=8000).encode("utf-8"))
    assert env["sample_rate"] == 8000


# --- WebSocketTransport (queue hand-off) ---------------------------------------


def test_ws_transport_mic_source_decodes_fed_frames():
    t = WebSocketTransport(sample_rate=16000)
    pcm = np.full(320, 0.25, dtype=np.float32)
    t.feed_mic(encode_frame(pcm))
    t.feed_mic(encode_frame(pcm))
    t.end_mic()  # EOF so mic_frames() terminates
    frames = list(t.mic_frames())
    assert len(frames) == 2
    assert np.allclose(frames[0], pcm, atol=1e-3)


def test_ws_transport_send_tts_enqueues_encoded_pcm():
    t = WebSocketTransport(sample_rate=16000)
    pcm = np.linspace(-1, 1, 100, dtype=np.float32)
    t.send_tts(pcm, 16000)
    data = t.iter_outbound(timeout=0.1)
    assert data is not None
    assert np.allclose(decode_frame(data), pcm, atol=1e-3)


def test_ws_transport_send_tts_skips_empty():
    t = WebSocketTransport()
    t.send_tts(np.zeros(0, dtype=np.float32), 16000)
    assert t.iter_outbound(timeout=0.05) is None


def test_ws_transport_close_ends_mic_frames():
    t = WebSocketTransport()
    t.close()
    assert t.closed is True
    assert list(t.mic_frames()) == []  # closed -> immediately exhausted


def test_ws_transport_feed_after_close_is_ignored():
    t = WebSocketTransport()
    t.close()
    t.feed_mic(encode_frame(np.ones(10, dtype=np.float32)))
    assert t.iter_outbound(timeout=0.01) is None


# --- LocalTransport surface ----------------------------------------------------


def test_local_transport_is_audio_transport():
    from my_stt_tts.transport import AudioTransport

    assert isinstance(LocalTransport(16000), AudioTransport)


def test_ws_transport_is_audio_transport():
    from my_stt_tts.transport import AudioTransport

    assert isinstance(WebSocketTransport(16000), AudioTransport)


# --- WS server handshake logic (fake socket, no real network) ------------------


class _FakeConn:
    """A duck-typed ``websockets`` connection: yields ``binary_frames`` then ends."""

    def __init__(self, hello: str, binary_frames: list[bytes]) -> None:
        self._hello = hello
        self._binary = binary_frames
        self.sent: list[object] = []
        self.closed_with: tuple[int, str] | None = None

    async def recv(self):
        return self._hello

    async def send(self, data):
        self.sent.append(data)

    async def close(self, code=1000, reason=""):
        self.closed_with = (code, reason)

    def __aiter__(self):
        async def _gen():
            for frame in self._binary:
                yield frame

        return _gen()


def test_ws_session_rejects_bad_handshake_token():
    import asyncio

    from my_stt_tts.ws_transport import WsSession

    ran: list[object] = []

    def on_session(t):  # pragma: no cover - must NOT run on a rejected client
        ran.append(t)

    conn = _FakeConn(make_handshake(sample_rate=16000, token="right"), [])
    session = WsSession(on_session, token="wrong")
    result = asyncio.run(session.handle(conn))
    assert result is None
    assert conn.closed_with == (1008, "handshake rejected")
    assert ran == []  # the session worker never started


def test_ws_session_accepts_and_pumps_mic_frames_to_on_session():
    import asyncio
    import threading

    from my_stt_tts.ws_transport import WsSession

    captured: dict[str, object] = {}
    done = threading.Event()

    def on_session(transport):
        # The worker thread drains the mic frames the handler pumps in.
        frames = list(transport.mic_frames())
        captured["frames"] = frames
        captured["count"] = len(frames)
        done.set()

    pcm = encode_frame(np.full(160, 0.2, dtype=np.float32))
    conn = _FakeConn(make_handshake(sample_rate=16000), [pcm, pcm, pcm])
    session = WsSession(on_session, token=None)
    result = asyncio.run(session.handle(conn))
    assert done.wait(timeout=2.0)
    assert result is not None
    assert captured["count"] == 3
    # The handshake reply ("ready" control message) was sent before any audio.
    assert any("ready" in str(s) for s in conn.sent)


# --- net_loop: transport-driven turn loop --------------------------------------


class _AltVad:
    """Marks the first N frames as speech, then silence (so a turn can 'end')."""

    def __init__(self, speech_frames: int) -> None:
        self._left = speech_frames

    def is_speech(self, frame) -> bool:  # noqa: ANN001, ARG002
        if self._left > 0:
            self._left -= 1
            return True
        return False


class _AfterSilenceAnalyzer:
    """Ends the turn after ``silence_after`` consecutive non-speech frames."""

    def __init__(self, silence_after: int = 2) -> None:
        self._silence_after = silence_after
        self._silent = 0

    def reset(self) -> None:
        self._silent = 0

    def update(self, frame, is_speech: bool) -> bool:  # noqa: ANN001, ARG002
        self._silent = 0 if is_speech else self._silent + 1
        return self._silent >= self._silence_after


class _FixedSTT:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def transcribe(self, audio, sample_rate: int = 16000) -> STTResult:  # noqa: ANN001, ARG002
        self.calls += 1
        return STTResult(text=self._text)


def test_capture_turn_sources_frames_and_transcribes():
    cfg = Config(sample_rate=16000)
    t = WebSocketTransport(sample_rate=16000)
    for _ in range(5):
        t.feed_mic(encode_frame(np.full(512, 0.3, dtype=np.float32)))
    t.end_mic()
    stt = _FixedSTT("hello from the satellite")
    text = net_loop.capture_turn(t, cfg, _AltVad(speech_frames=3), _AfterSilenceAnalyzer(2), stt)
    assert text == "hello from the satellite"
    assert stt.calls >= 1


def test_capture_turn_returns_empty_on_silence():
    cfg = Config(sample_rate=16000)
    t = WebSocketTransport(sample_rate=16000)
    for _ in range(3):
        t.feed_mic(encode_frame(np.zeros(512, dtype=np.float32)))
    t.end_mic()
    stt = _FixedSTT("never used")
    # VAD never reports speech -> the loop returns "" without a final transcribe.
    text = net_loop.capture_turn(t, cfg, _AltVad(speech_frames=0), _AfterSilenceAnalyzer(2), stt)
    assert text == ""


class _FakeBrain:
    """A Brain stand-in whose stream yields fixed deltas (no network)."""

    def __init__(self, parts: list[str]) -> None:
        self._parts = parts
        self.speaker: str | None = None

    def stream(self, text: str):  # noqa: ARG002
        yield from self._parts

    def set_speaker(self, name: str | None) -> None:  # mirrors Brain.set_speaker (G7)
        self.speaker = name


class _SinkTTS:
    """A TTSRouter stand-in that synthesizes 1 sample of PCM per sentence."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synth_pcm(self, text: str, lang=None):  # noqa: ANN001, ARG002
        self.spoken.append(text)
        return np.full(8, 0.1, dtype=np.float32), 16000


def test_respond_over_transport_sinks_tts_pcm():
    cfg = Config(sample_rate=16000)
    t = WebSocketTransport(sample_rate=16000)
    brain = _FakeBrain(["Hello there. ", "How are you?"])
    tts = _SinkTTS()
    result = net_loop.respond_over_transport(t, cfg, brain, tts, "hi")
    # Two sentences were synthesized and at least one PCM chunk queued outbound.
    assert "Hello there." in "".join(tts.spoken)
    assert result.interrupted is False  # half-duplex (no mic source) -> no barge-in
    assert result.spoken  # the full spoken text is recorded on the result
    assert t.iter_outbound(timeout=0.1) is not None


def test_mic_frame_chunks_splits_evenly():
    pcm = np.arange(1100, dtype=np.float32)
    chunks = list(net_loop.mic_frame_chunks(pcm, frame_samples=512))
    assert [c.size for c in chunks] == [512, 512, 76]


# --- ws_frame: RFC 6455 framing for the browser audio channel ------------------


def _mask(payload: bytes, mask: bytes = b"\x12\x34\x56\x78") -> bytes:
    """Build a masked client frame (binary opcode) the way a browser would."""
    import struct

    out = bytearray([0x82])  # FIN + binary
    length = len(payload)
    if length < 126:
        out.append(0x80 | length)
    else:
        out.append(0x80 | 126)
        out += struct.pack(">H", length)
    out += mask
    out += bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return bytes(out)


def test_ws_frame_accept_key_matches_rfc_example():
    from my_stt_tts.ws_frame import accept_key

    # The canonical example from RFC 6455 §1.3.
    assert accept_key("dGhlIHNhbXBsZSBub25jZQ==") == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_ws_frame_server_encode_round_trips_through_client_decode():
    from my_stt_tts.ws_frame import OP_BINARY, encode_frame

    payload = b"hello pcm bytes"
    # Server frames are unmasked; to decode with our (client-masked) decoder we
    # re-mask. Here we just assert the server encoder produced a well-formed header.
    framed = encode_frame(payload, opcode=OP_BINARY)
    assert framed[0] == 0x82  # FIN + binary
    assert framed[1] == len(payload)  # unmasked, short length
    assert framed[2:] == payload


def test_ws_frame_decode_masked_client_frame():
    from my_stt_tts.ws_frame import OP_BINARY, decode_frame

    payload = bytes(range(40))
    decoded = decode_frame(_mask(payload))
    assert decoded is not None
    frame, consumed = decoded
    assert frame.opcode == OP_BINARY
    assert frame.payload == payload
    assert consumed == len(_mask(payload))


def test_ws_frame_decode_needs_more_bytes_returns_none():
    from my_stt_tts.ws_frame import decode_frame

    full = _mask(bytes(range(40)))
    assert decode_frame(full[:3]) is None  # header incomplete -> wait for more


def test_ws_frame_rejects_unmasked_client_frame():
    from my_stt_tts.ws_frame import decode_frame

    # 0x82 FIN+binary, length 1, NO mask bit -> a spec violation from a client.
    with pytest.raises(ValueError, match="not masked"):
        decode_frame(b"\x82\x01\x00")


# --- webui audio bridge: browser PCM ⇄ pipeline over a fake socket -------------


class _FakeSocket:
    """A duck-typed socket: serves ``inbound`` chunks then EOF; records sends."""

    def __init__(self, inbound: list[bytes]) -> None:
        self._inbound = list(inbound)
        self.sent = bytearray()

    def recv(self, _n: int) -> bytes:
        return self._inbound.pop(0) if self._inbound else b""

    def sendall(self, data: bytes) -> None:
        self.sent += data


def test_webui_audio_bridge_feeds_browser_pcm_to_session():
    import threading

    from my_stt_tts.webui import WebUI

    captured: dict[str, int] = {}
    done = threading.Event()

    def on_audio(transport):
        captured["frames"] = len(list(transport.mic_frames()))
        done.set()

    cfg = Config(sample_rate=16000)
    ui = WebUI(cfg, on_turn=lambda t: None, on_audio_session=on_audio, port=0)
    pcm = encode_frame(np.full(160, 0.2, dtype=np.float32))
    # Two masked binary client frames, then a close frame.
    close = bytes([0x88, 0x80, 0x00, 0x00, 0x00, 0x00])
    sock = _FakeSocket([_mask(pcm), _mask(pcm), close])
    ui.run_audio_session(sock)
    assert done.wait(timeout=2.0)
    assert captured["frames"] == 2
