"""Tests for the round-3 breadth/ops gaps (R3-5/7/8/9):

* R3-5 — speech-to-speech / realtime LLM: the WS event protocol (pure), the
  base64-PCM transcode, and the mic→endpoint→audio-back round-trip against a
  MOCKED realtime WS server (no key, no network).
* R3-7 — per-stage latency telemetry: the fake-clock TurnMetrics record, the
  aggregator (count/mean/p50/p95), the JSON-lines log, and the bus emit.
* R3-8 — verified first-run bootstrap: the checksum verify, the corrupt-download
  rejection, the preflight report (happy / missing / corrupt), and the runtime
  silence-fallback warning (log + bus event + on_fallback hook).
* R3-9 — telephony reach: the G.711 μ-law transcode + 8k/16k resample, the Twilio
  Media Streams event protocol, and the per-call session over a fake socket.

Everything fakes the network / provider / clock / socket / download boundaries —
nothing here opens a socket, downloads a file, calls an API, or reads a real clock.
"""
# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring
# pylint: disable=too-few-public-methods,redefined-outer-name,import-outside-toplevel

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from my_stt_tts.config import Config

# ============================================================================
# R3-7 — per-stage latency telemetry
# ============================================================================


class _FakeClock:
    """A monotonically-advanceable clock so timings are deterministic (no real time)."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def test_turn_metrics_marks_stages_with_fake_clock():
    from my_stt_tts.metrics import TurnMetrics

    clk = _FakeClock()
    m = TurnMetrics(speech_id="turn-x", clock=clk)
    clk.advance(0.08)
    m.mark("stt")
    clk.advance(0.12)
    m.mark("llm_first_token")
    clk.advance(0.15)
    m.mark("first_audio")
    d = m.as_dict()
    assert d["speech_id"] == "turn-x"
    assert d["stages_ms"] == {"stt": 80.0, "llm_first_token": 200.0, "first_audio": 350.0}
    assert d["total_ms"] == 350.0


def test_turn_metrics_stage_context_manager_times_a_span():
    from my_stt_tts.metrics import TurnMetrics

    clk = _FakeClock()
    m = TurnMetrics(clock=clk)
    with m.stage("tts"):
        clk.advance(0.25)
    assert m.stages["tts"] == 250.0


def test_metrics_aggregator_count_mean_percentiles():
    from my_stt_tts.metrics import MetricsAggregator, TurnMetrics

    agg = MetricsAggregator()
    for ms in (100.0, 200.0, 300.0, 400.0):
        clk = _FakeClock()
        m = TurnMetrics(clock=clk)
        clk.advance(ms / 1000.0)
        m.mark("first_audio")
        agg.add(m)
    assert agg.count == 4
    s = agg.summary()["first_audio"]
    assert s["count"] == 4.0
    assert s["mean"] == 250.0
    assert s["min"] == 100.0
    assert s["max"] == 400.0
    assert 200.0 <= s["p50"] <= 300.0
    assert s["p95"] >= 350.0


def test_metrics_log_writes_one_json_line_per_turn(tmp_path):
    from my_stt_tts.metrics import MetricsLog

    path = tmp_path / "metrics.jsonl"
    mlog = MetricsLog(path)
    mlog.write({"speech_id": "t1", "stages_ms": {"stt": 50.0}})
    mlog.write({"speech_id": "t2", "stages_ms": {"stt": 60.0}})
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["speech_id"] == "t1"
    assert json.loads(lines[1])["stages_ms"]["stt"] == 60.0


def test_turn_metrics_emit_publishes_to_bus_and_sink(tmp_path):
    from my_stt_tts import events
    from my_stt_tts.metrics import TelemetrySink, TurnMetrics

    sub = events.bus.subscribe()
    try:
        sink = TelemetrySink(log_file=tmp_path / "m.jsonl")
        clk = _FakeClock()
        m = TurnMetrics(speech_id="turn-emit", clock=clk)
        clk.advance(0.1)
        m.mark("first_audio")
        record = m.emit(sink)
        # The full turn made it to the JSON-lines file + the aggregator.
        assert sink.aggregator.count == 1
        assert "first_audio" in sink.summary()
        assert (tmp_path / "m.jsonl").exists()
        # A `metrics` event was published to the bus.
        seen = []
        while not sub.empty():
            seen.append(json.loads(sub.get_nowait()))
        assert any(e.get("type") == "metrics" and e.get("speech_id") == "turn-emit" for e in seen)
        assert record["speech_id"] == "turn-emit"
    finally:
        events.bus.unsubscribe(sub)


def test_make_sink_off_by_default_and_on_when_enabled(tmp_path):
    from my_stt_tts.metrics import make_sink

    assert make_sink(Config(telemetry=False)) is None
    sink = make_sink(Config(telemetry=True, telemetry_log_file=str(tmp_path / "x.jsonl")))
    assert sink is not None


# ============================================================================
# R3-9 — telephony reach (Twilio Media Streams, G.711 μ-law)
# ============================================================================


def test_ulaw_round_trip_is_near_lossless_on_speech():
    from my_stt_tts.telephony import ulaw_decode, ulaw_encode

    t = np.linspace(0, 0.25, 2000)
    sig = (np.sin(2 * np.pi * 300 * t) * 15000).astype(np.int16)
    rt = ulaw_decode(ulaw_encode(sig)).astype(np.float64)
    rms = np.sqrt(np.mean((sig.astype(np.float64) - rt) ** 2))
    # μ-law is logarithmic; round-trip RMS error stays a small fraction of amplitude.
    assert rms < 1500.0


def test_ulaw_decode_matches_standard_levels():
    # Standard G.711: byte 0xFF (silence) decodes to ~0; 0x00 to the max negative.
    from my_stt_tts.telephony import ulaw_decode

    levels = ulaw_decode(bytes(range(256)))
    assert abs(int(levels[0xFF])) <= 8  # near silence
    assert int(levels[0x00]) < -30000  # max-magnitude negative
    assert int(levels[0x80]) > 30000  # max-magnitude positive


def test_ulaw_encode_is_nearest_quantizer():
    # decode(encode(x)) must be the closest representable μ-law level to x.
    from my_stt_tts.telephony import ulaw_decode, ulaw_encode

    levels = ulaw_decode(bytes(range(256))).astype(np.int32)
    xs = np.array([-30000, -1000, -8, 0, 7, 1000, 30000], dtype=np.int16)
    rt = ulaw_decode(ulaw_encode(xs)).astype(np.int32)
    nearest = levels[np.abs(xs.astype(np.int32)[:, None] - levels[None, :]).argmin(axis=1)]
    assert np.array_equal(rt, nearest)


def test_ulaw_empty_inputs():
    from my_stt_tts.telephony import ulaw_decode, ulaw_encode

    assert ulaw_encode(np.zeros(0, dtype=np.int16)) == b""
    assert ulaw_decode(b"").size == 0


def test_resample_8k_16k_doubles_and_back():
    from my_stt_tts.telephony import resample_8k_16k, resample_16k_8k

    src = np.full(160, 0.3, dtype=np.float32)  # one 8 kHz frame (20 ms)
    up = resample_8k_16k(src)
    assert up.size == 320  # 8k -> 16k doubles the sample count
    down = resample_16k_8k(up)
    assert down.size == 160
    assert np.allclose(down, 0.3, atol=1e-2)


def test_twilio_serializer_decodes_start_then_media_to_pcm():
    from my_stt_tts.telephony import (
        TwilioMediaStreamSerializer,
        pcm_float_to_int16,
        ulaw_encode,
    )

    s = TwilioMediaStreamSerializer(sample_rate=16000)
    start = s.decode(
        json.dumps({"event": "start", "start": {"streamSid": "MZ1", "callSid": "CA1"}})
    )
    assert start == {"event": "start", "stream_sid": "MZ1", "call_sid": "CA1"}
    assert s.stream_sid == "MZ1"
    # A Twilio inbound media frame: base64(μ-law(8 kHz PCM)).
    src8k = np.full(160, 0.25, dtype=np.float32)
    import base64

    payload = base64.b64encode(ulaw_encode(pcm_float_to_int16(src8k))).decode()
    media = s.decode(json.dumps({"event": "media", "media": {"payload": payload}}))
    assert media["event"] == "media"
    # Decoded + upsampled to 16 kHz (160 -> 320) with the value preserved.
    assert media["pcm"].size == 320
    assert np.allclose(media["pcm"], 0.25, atol=2e-2)


def test_twilio_serializer_encode_media_keyed_by_stream_sid():
    from my_stt_tts.telephony import TwilioMediaStreamSerializer

    s = TwilioMediaStreamSerializer(sample_rate=16000)
    # Before `start` there is no streamSid -> nothing to send.
    assert s.encode_media(np.full(320, 0.3, dtype=np.float32)) is None
    s.decode(json.dumps({"event": "start", "start": {"streamSid": "MZ9"}}))
    frame = s.encode_media(np.full(320, 0.3, dtype=np.float32))
    assert frame is not None
    env = json.loads(frame)
    assert env["event"] == "media"
    assert env["streamSid"] == "MZ9"
    assert env["media"]["payload"]  # non-empty base64 μ-law


def test_twilio_serializer_handles_junk_and_stop():
    from my_stt_tts.telephony import TwilioMediaStreamSerializer

    s = TwilioMediaStreamSerializer()
    assert s.decode("not json")["event"] == "unknown"
    assert s.decode(json.dumps([1, 2, 3]))["event"] == "unknown"
    assert s.decode(json.dumps({"event": "stop"}))["event"] == "stop"
    assert s.stopped is True


def test_twilio_transport_is_audio_transport_and_round_trips():
    from my_stt_tts.telephony import TwilioMediaStreamSerializer, TwilioTransport
    from my_stt_tts.transport import AudioTransport

    s = TwilioMediaStreamSerializer(sample_rate=16000)
    s.decode(json.dumps({"event": "start", "start": {"streamSid": "MZ"}}))
    t = TwilioTransport(s)
    assert isinstance(t, AudioTransport)
    # mic source side
    t.feed_mic(np.full(320, 0.1, dtype=np.float32))
    t.end_mic()
    frames = list(t.mic_frames())
    assert len(frames) == 1 and frames[0].size == 320
    # TTS sink side -> an outbound Twilio media frame
    t.send_tts(np.full(320, 0.2, dtype=np.float32), 16000)
    out = t.iter_outbound(timeout=0.2)
    assert out is not None
    assert json.loads(out)["event"] == "media"


class _FakeTwilioConn:
    """A duck-typed Twilio WS connection: yields scripted frames, records sends."""

    def __init__(self, frames: list[str]) -> None:
        self._frames = frames
        self.sent: list[str] = []

    async def send(self, data) -> None:  # noqa: ANN001
        self.sent.append(data)

    def __aiter__(self):
        async def _gen():
            import asyncio

            await asyncio.sleep(0.05)  # let the worker thread start
            for f in self._frames:
                yield f

        return _gen()


def test_twilio_session_bridges_a_call_to_on_session():
    import asyncio
    import base64
    import threading

    from my_stt_tts.telephony import TwilioSession, pcm_float_to_int16, ulaw_encode

    captured: dict[str, int] = {}
    done = threading.Event()

    def on_session(transport):
        frames = list(transport.mic_frames())
        captured["frames"] = len(frames)
        done.set()

    payload = base64.b64encode(
        ulaw_encode(pcm_float_to_int16(np.full(160, 0.2, dtype=np.float32)))
    ).decode()
    frames = [
        json.dumps({"event": "connected"}),
        json.dumps({"event": "start", "start": {"streamSid": "MZ"}}),
        json.dumps({"event": "media", "media": {"payload": payload}}),
        json.dumps({"event": "media", "media": {"payload": payload}}),
        json.dumps({"event": "stop"}),
    ]
    conn = _FakeTwilioConn(frames)
    session = TwilioSession(on_session, sample_rate=16000)
    asyncio.run(session.handle(conn))
    assert done.wait(timeout=2.0)
    assert captured["frames"] == 2  # the two media frames were decoded + fed


# ============================================================================
# R3-8 — verified first-run bootstrap (checksum + preflight + fallback)
# ============================================================================


def test_verify_checksum_good_bad_missing_and_empty_pin(tmp_path):
    from my_stt_tts.turn import verify_checksum

    f = tmp_path / "m.bin"
    f.write_bytes(b"hello world")
    good = hashlib.sha256(b"hello world").hexdigest()
    assert verify_checksum(f, good) is True
    assert verify_checksum(f, good.upper()) is True  # case-insensitive
    assert verify_checksum(f, "deadbeef") is False
    assert verify_checksum(f, "") is True  # no pin -> skip
    assert verify_checksum(tmp_path / "missing", good) is False


def test_ensure_smart_turn_rejects_corrupt_download(tmp_path, monkeypatch):
    import my_stt_tts.turn as turnmod

    target = tmp_path / "dl.onnx"
    expected = hashlib.sha256(b"the real model bytes").hexdigest()

    class _Resp:
        def read(self):
            return b"CORRUPT (truncated) bytes"  # wrong content -> hash mismatch

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(turnmod.urllib.request, "urlopen", lambda *a, **k: _Resp())
    ok = turnmod.ensure_smart_turn_model(
        str(target), "https://x/y", auto_download=True, expected_sha256=expected
    )
    assert ok is False  # checksum mismatch -> not installed
    assert not target.exists()  # corrupt file deleted, not left behind


def test_ensure_smart_turn_accepts_matching_download(tmp_path, monkeypatch):
    import my_stt_tts.turn as turnmod

    target = tmp_path / "dl.onnx"
    payload = b"the real model bytes"
    expected = hashlib.sha256(payload).hexdigest()

    class _Resp:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(turnmod.urllib.request, "urlopen", lambda *a, **k: _Resp())
    ok = turnmod.ensure_smart_turn_model(
        str(target), "https://x/y", auto_download=True, expected_sha256=expected
    )
    assert ok is True
    assert turnmod.file_sha256(target) == expected


def test_ensure_smart_turn_redownloads_present_but_corrupt_file(tmp_path, monkeypatch):
    import my_stt_tts.turn as turnmod

    target = tmp_path / "dl.onnx"
    target.write_bytes(b"stale corrupt cache")  # present but wrong
    payload = b"good model"
    expected = hashlib.sha256(payload).hexdigest()

    class _Resp:
        def read(self):
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(turnmod.urllib.request, "urlopen", lambda *a, **k: _Resp())
    ok = turnmod.ensure_smart_turn_model(
        str(target), "https://x/y", auto_download=True, expected_sha256=expected
    )
    assert ok is True
    assert target.read_bytes() == payload  # the bad cache was replaced


def test_smart_turn_analyzer_warns_once_on_silence_fallback():
    from my_stt_tts import events
    from my_stt_tts.turn import SmartTurnAnalyzer

    reasons: list[str] = []
    sub = events.bus.subscribe()
    try:
        a = SmartTurnAnalyzer(
            "/nonexistent/model.onnx",
            silence_seconds=0.05,
            frame_seconds=0.032,
            auto_download=False,  # genuinely unavailable -> must fall back
            on_fallback=reasons.append,
        )
        # Speak briefly, then go silent so the candidate endpointer arms + fires.
        ended = any(a.update(0.1 if i < 3 else 0.0, is_speech=i < 3) for i in range(20))
        assert ended is True  # falls back to plain silence endpointing
        assert len(reasons) == 1  # the warning hook fired exactly once
        # And a structured bus event was published for the UI.
        events_seen = []
        while not sub.empty():
            events_seen.append(json.loads(sub.get_nowait()))
        assert any(e.get("type") == "endpoint_fallback" for e in events_seen)
    finally:
        events.bus.unsubscribe(sub)


def test_preflight_report_happy_missing_and_corrupt():
    from my_stt_tts.preflight import run_preflight

    cfg = Config()
    # Happy path: model + voices all fetch + verify.
    rep = run_preflight(
        cfg,
        ensure_model=lambda p, u, a, s: True,
        ensure_voice=lambda d, v: True,
        checksum=lambda p, s: True,
    )
    assert rep.ok is True
    assert "all components ready" in rep.text()

    # Download succeeds but checksum fails -> not ready, clearly flagged.
    rep2 = run_preflight(
        cfg,
        ensure_model=lambda p, u, a, s: True,
        ensure_voice=lambda d, v: True,
        checksum=lambda p, s: False,
    )
    assert rep2.ok is False
    assert "MISMATCH" in rep2.text()

    # Download fails outright -> not ready.
    rep3 = run_preflight(
        cfg,
        ensure_model=lambda p, u, a, s: False,
        ensure_voice=lambda d, v: True,
        checksum=lambda p, s: True,
    )
    assert rep3.ok is False


def test_preflight_skips_smart_turn_when_silence_analyzer():
    from my_stt_tts.preflight import check_smart_turn

    cfg = Config(turn_analyzer="silence")
    called = []
    res = check_smart_turn(cfg, ensure=lambda *a: called.append(a) or True)
    assert res.ok is True
    assert "not needed" in res.detail
    assert not called  # no download attempted


# ============================================================================
# R3-5 — speech-to-speech / realtime LLM
# ============================================================================


def test_realtime_pcm_base64_round_trip():
    from my_stt_tts.realtime import base64_to_pcm, pcm_to_base64

    pcm = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    out = base64_to_pcm(pcm_to_base64(pcm))
    assert np.allclose(out, pcm, atol=1e-3)
    assert pcm_to_base64(np.zeros(0, dtype=np.float32)) == ""
    assert base64_to_pcm("").size == 0


def test_realtime_protocol_builds_client_events():
    from my_stt_tts.realtime import RealtimeProtocol

    p = RealtimeProtocol(voice="verse", audio_format="pcm16", instructions="be brief")
    su = json.loads(p.session_update())
    assert su["type"] == "session.update"
    assert su["session"]["voice"] == "verse"
    assert su["session"]["input_audio_format"] == "pcm16"
    assert su["session"]["turn_detection"]["type"] == "server_vad"
    assert su["session"]["instructions"] == "be brief"
    ap = json.loads(p.append_audio(np.full(160, 0.2, dtype=np.float32)))
    assert ap["type"] == "input_audio_buffer.append"
    assert ap["audio"]  # base64 payload
    assert json.loads(p.commit_audio())["type"] == "input_audio_buffer.commit"
    assert json.loads(p.create_response())["type"] == "response.create"
    assert json.loads(p.cancel_response())["type"] == "response.cancel"


def test_realtime_protocol_decodes_server_events():
    from my_stt_tts.realtime import RealtimeProtocol, pcm_to_base64

    p = RealtimeProtocol()
    audio_b64 = pcm_to_base64(np.full(240, 0.3, dtype=np.float32))
    delta = p.decode(json.dumps({"type": "response.audio.delta", "delta": audio_b64}))
    assert delta["type"] == "response.audio.delta"
    assert delta["pcm"].size == 240
    txt = p.decode(json.dumps({"type": "response.audio_transcript.delta", "delta": "hi"}))
    assert txt["text"] == "hi"
    done = p.decode(json.dumps({"type": "response.audio_transcript.done", "transcript": "hello"}))
    assert done["text"] == "hello"
    err = p.decode(json.dumps({"type": "error", "error": {"message": "boom"}}))
    assert err == {"type": "error", "message": "boom"}
    assert p.decode("garbage")["type"] == "unknown"


def test_make_realtime_brain_gates_on_key():
    from my_stt_tts.realtime import make_realtime_brain

    # Not selected -> None (cascade).
    assert make_realtime_brain(Config(brain_mode="cascade")) is None
    # Selected but no key -> None (graceful fallback).
    assert make_realtime_brain(Config(brain_mode="realtime", realtime_api_key=None)) is None
    # Selected AND keyed -> a RealtimeBrain.
    brain = make_realtime_brain(Config(brain_mode="realtime", realtime_api_key="sk-test"))
    assert brain is not None and brain.available() is True


class _FakeRealtimeConn:
    """A mocked OpenAI Realtime WS server: records sends, yields scripted events."""

    def __init__(self, server_events: list[str], *, delay: float = 0.0) -> None:
        self._events = server_events
        self._delay = delay
        self.sent: list[str] = []
        self.closed = False

    async def send(self, data) -> None:  # noqa: ANN001
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True

    def __aiter__(self):
        async def _gen():
            import asyncio

            if self._delay:
                await asyncio.sleep(self._delay)
            for e in self._events:
                yield e

        return _gen()


def test_run_realtime_session_round_trip_over_mocked_ws():
    from my_stt_tts.realtime import REALTIME_SR, pcm_to_base64, run_realtime_session
    from my_stt_tts.transport import encode_frame
    from my_stt_tts.ws_transport import WebSocketTransport

    # A transport whose mic yields a couple of frames; record what TTS gets sunk.
    t = WebSocketTransport(sample_rate=16000)
    for _ in range(3):
        t.feed_mic(encode_frame(np.full(160, 0.2, dtype=np.float32)))
    t.end_mic()
    sunk: list[tuple[int, int]] = []

    def _record(pcm, sr):  # noqa: ANN001, ANN202
        sunk.append((np.asarray(pcm).size, sr))

    t.send_tts = _record  # type: ignore[method-assign]

    reply = pcm_to_base64(np.full(240, 0.3, dtype=np.float32))
    events = [
        json.dumps({"type": "response.audio_transcript.delta", "delta": "Hello"}),
        json.dumps({"type": "response.audio.delta", "delta": reply}),
        json.dumps({"type": "response.audio_transcript.done", "transcript": "Hello there"}),
        json.dumps({"type": "response.done"}),
    ]
    # The small delay lets the mic-pump thread send appends before the server replies.
    conn = _FakeRealtimeConn(events, delay=0.15)
    cfg = Config(realtime_api_key="sk-test")
    run_realtime_session(t, cfg, connection=conn, max_turns=1)

    # session.update was the first thing sent, mic audio was appended, the reply
    # audio was sunk back to the transport (at the realtime 24 kHz rate), and the
    # connection was closed cleanly.
    assert "session.update" in conn.sent[0]
    assert sum("input_audio_buffer.append" in s for s in conn.sent) == 3
    assert sunk == [(240, REALTIME_SR)]
    assert conn.closed is True


def test_realtime_client_connect_requires_key():
    from my_stt_tts.realtime import RealtimeClient, RealtimeError

    client = RealtimeClient(Config(realtime_api_key=None))
    assert client.available() is False
    import asyncio

    with pytest.raises(RealtimeError, match="REALTIME_API_KEY"):
        asyncio.run(client.connect())


# ============================================================================
# config wiring for the new gaps
# ============================================================================


def test_config_validates_new_modes():
    from my_stt_tts.config import ConfigError

    Config(brain_mode="realtime", realtime_api_key="k", anthropic_api_key="a").validate()
    with pytest.raises(ConfigError, match="brain_mode"):
        Config(brain_mode="bogus", anthropic_api_key="a").validate()
    with pytest.raises(ConfigError, match="realtime_audio_format"):
        Config(realtime_audio_format="flac", anthropic_api_key="a").validate()
    with pytest.raises(ConfigError, match="telephony_port"):
        Config(telephony_port=0, anthropic_api_key="a").validate()


def test_config_from_env_reads_new_vars(monkeypatch):
    monkeypatch.setenv("BRAIN_MODE", "realtime")
    monkeypatch.setenv("REALTIME_API_KEY", "sk-rt")
    monkeypatch.setenv("TELEMETRY", "true")
    monkeypatch.setenv("TELEMETRY_LOG_FILE", "/tmp/m.jsonl")
    monkeypatch.setenv("TELEPHONY", "1")
    monkeypatch.setenv("TELEPHONY_PORT", "9999")
    cfg = Config.from_env()
    assert cfg.brain_mode == "realtime"
    assert cfg.realtime_api_key == "sk-rt"
    assert cfg.telemetry is True
    assert cfg.telemetry_log_file == "/tmp/m.jsonl"
    assert cfg.telephony is True
    assert cfg.telephony_port == 9999


def test_smart_turn_sha256_pin_is_set():
    # The integrity pin ships by default so a corrupt download is caught out of the box.
    cfg = Config()
    assert len(cfg.smart_turn_sha256) == 64  # a full SHA-256 hex digest
    assert Path(cfg.smart_turn_model_path).name == "smart-turn-v3.0.onnx"
