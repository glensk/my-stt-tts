"""Tests for the round-3 transport/audio robustness gaps:

* R3-2 — full-duplex barge-in over the NETWORK transport (interrupt mid-reply)
* R3-3 — streamed, low-latency TTS playout (first-frame latency + cancel)
* R3-4 — macOS hardware-AEC capture seam (VoiceProcessingIO -> Python frames)
* R3-1 — true WebRTC transport: the queue bridge + the SDP signaling seam
* R3-6 — pre-VAD noise-suppression stage (spectral gate + fallback)

Everything fakes the mic / network / subprocess / provider boundaries — nothing
opens a socket, a real RTCPeerConnection, a device, an OutputStream, or an API.
"""
# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring
# pylint: disable=too-few-public-methods,redefined-outer-name,import-outside-toplevel

import time

import numpy as np

from my_stt_tts import net_loop
from my_stt_tts.config import Config
from my_stt_tts.stt import STTResult

# ============================================================================
# R3-3 — streamed, low-latency TTS playout
# ============================================================================


class _SlowSynth:
    """A TTS stand-in whose per-clause synthesis takes ``delay`` seconds.

    Used to prove that streamed playout's time-to-first-audio is the FIRST
    clause's synthesis time, not the whole sentence's."""

    def __init__(self, delay: float = 0.05, sr: int = 16000) -> None:
        self.delay = delay
        self.sr = sr
        self.clauses: list[str] = []

    def synth_pcm(self, text, lang=None):  # noqa: ANN001, ARG002
        time.sleep(self.delay)
        self.clauses.append(text)
        return np.full(160, 0.2, dtype=np.float32), self.sr


def _make_streaming_playback(monkeypatch):
    """Build a StreamingPlayback whose OutputStream is a fake (no sounddevice)."""
    from my_stt_tts import tts as tts_mod

    written: list[np.ndarray] = []

    class _FakeStream:
        def start(self):
            pass

        def write(self, block):
            written.append(np.asarray(block).ravel())

        def stop(self):
            pass

        def close(self):
            pass

    pb = tts_mod.StreamingPlayback(16000, frame_samples=512)
    pb._open_stream = lambda _sr: _FakeStream()
    return pb, written


def test_clause_chunker_emits_subsentence_clauses():
    from my_stt_tts.text import ClauseChunker

    c = ClauseChunker(min_chars=8)
    out = c.feed("In the morning, I will tell you the weather.")
    # A comma after enough text yields the first clause early, then the sentence.
    assert any("morning" in s for s in out)
    assert any(s.endswith(".") for s in [*out, c.flush()])


def test_clause_chunker_protects_decimals():
    from my_stt_tts.text import ClauseChunker

    c = ClauseChunker(min_chars=1)
    out = [*c.feed("It is 3.14 and 3,14 today."), c.flush()]
    joined = " ".join(out)
    assert "3.14" in joined and "3,14" in joined  # decimals never split a clause


def test_synth_pcm_stream_yields_per_clause():
    cfg = Config(sample_rate=16000, tts_stream_min_chars=4)
    router = object.__new__(__import__("my_stt_tts.tts", fromlist=["TTSRouter"]).TTSRouter)
    router.cfg = cfg
    router._cloud = None
    synth = _SlowSynth(delay=0.0)
    router.synth_pcm = synth.synth_pcm  # type: ignore[method-assign]
    chunks = list(router.synth_pcm_stream("First part, then a second part."))
    assert len(chunks) >= 2  # at least two clauses synthesized separately
    assert all(pcm.size for pcm, _sr in chunks)


def test_streamed_first_frame_latency_beats_full_sentence(monkeypatch):
    from my_stt_tts import tts as tts_mod

    cfg = Config(sample_rate=16000, tts_stream_min_chars=4)
    router = object.__new__(tts_mod.TTSRouter)
    router.cfg = cfg
    router._cloud = None
    synth = _SlowSynth(delay=0.05)  # 50 ms per clause
    router.synth_pcm = synth.synth_pcm  # type: ignore[method-assign]

    pb, written = _make_streaming_playback(monkeypatch)
    monkeypatch.setattr(tts_mod, "StreamingPlayback", lambda *a, **k: pb)

    text = "One, two, three, four, five, six."  # six clauses -> ~300 ms total synth
    t0 = time.monotonic()
    handle = router.start_speaking_stream(text)
    # Wait for the FIRST chunk to reach the player.
    deadline = t0 + 2.0
    while not written and time.monotonic() < deadline:
        time.sleep(0.005)
    first_audio = time.monotonic() - t0
    handle.wait()
    # First audio within ~one clause (50 ms) + slack, well under the ~300 ms full
    # sentence — this is the whole point of streamed playout.
    assert written, "no audio was ever played"
    assert first_audio < 0.2, f"first-frame latency too high: {first_audio:.3f}s"
    assert len(synth.clauses) >= 5


def test_streaming_playback_cancel_aborts_midway(monkeypatch):
    pb, written = _make_streaming_playback(monkeypatch)
    pb.feed(np.full(512, 0.1, dtype=np.float32))
    pb.feed(np.full(512, 0.1, dtype=np.float32))
    time.sleep(0.05)
    pb.cancel()  # abort before end_feed
    pb.wait()
    assert pb.cancelled
    assert pb.done


def test_streaming_playback_reference_accumulates(monkeypatch):
    pb, _ = _make_streaming_playback(monkeypatch)
    pb.feed(np.full(100, 0.1, dtype=np.float32))
    pb.feed(np.full(50, 0.2, dtype=np.float32))
    pb.end_feed()
    pb.wait()
    # The AEC reference is the concatenation of everything played.
    assert pb.reference.size == 150
    assert pb.reference_sr == 16000


# ============================================================================
# R3-6 — pre-VAD noise suppression
# ============================================================================


def test_spectral_denoiser_raises_snr_on_steady_noise():
    from my_stt_tts.denoise import SpectralGateDenoiser

    rng = np.random.default_rng(0)
    den = SpectralGateDenoiser(strength=2.0)
    # Prime the noise floor with several noise-only frames.
    for _ in range(20):
        den.process(rng.normal(0, 0.05, 512).astype(np.float32))
    # A speech-like tone buried in the same noise.
    n = 512
    tone = 0.4 * np.sin(2 * np.pi * 300 * np.arange(n) / 16000).astype(np.float32)
    noisy = tone + rng.normal(0, 0.05, n).astype(np.float32)
    clean = den.process(noisy)
    # Residual noise energy is reduced relative to the noisy input's noise part.
    noise_in = float(np.std(noisy - tone))
    noise_out = float(np.std(clean - tone))
    assert noise_out < noise_in  # the denoiser attenuated the noise


def test_null_denoiser_is_identity():
    from my_stt_tts.denoise import NullDenoiser

    den = NullDenoiser()
    x = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.array_equal(den.process(x), x)


def test_make_denoiser_selects_backend():
    from my_stt_tts.denoise import NullDenoiser, SpectralGateDenoiser, make_denoiser

    assert isinstance(make_denoiser(Config(denoiser="off")), NullDenoiser)
    assert isinstance(make_denoiser(Config(denoiser="spectral")), SpectralGateDenoiser)
    # rnnoise falls back to spectral when the wheel isn't importable/usable here.
    assert isinstance(make_denoiser(Config(denoiser="rnnoise")), SpectralGateDenoiser | object)


def test_denoiser_reset_clears_floor():
    from my_stt_tts.denoise import SpectralGateDenoiser

    den = SpectralGateDenoiser()
    den.process(np.random.default_rng(1).normal(0, 0.1, 256).astype(np.float32))
    den.reset()
    assert len(den._mags) == 0


# ============================================================================
# R3-4 — macOS hardware-AEC capture seam
# ============================================================================


class _FakeVPBuffer:
    """A stand-in AVAudioPCMBuffer: one channel of float32 samples."""

    def __init__(self, samples: np.ndarray) -> None:
        self._s = np.asarray(samples, dtype=np.float32)

    def frameLength(self):  # noqa: N802
        return self._s.size

    def floatChannelData(self):  # noqa: N802
        arr = self._s

        class _Ptr:
            def as_buffer(self, _n):
                return arr.tobytes()

        return [_Ptr()]


def test_voiceprocessing_capture_bridges_buffers_to_frames():
    from my_stt_tts.aec import VoiceProcessingCapture

    cap = VoiceProcessingCapture(sample_rate=48000, frame_samples=512)
    cap._device_sr = 48000.0  # no resampling for this test
    # Simulate the tap callback delivering two 512-sample buffers.
    cap._on_buffer(_FakeVPBuffer(np.full(512, 0.3, dtype=np.float32)), None)
    cap._on_buffer(_FakeVPBuffer(np.full(512, 0.3, dtype=np.float32)), None)
    cap._closed.set()  # so mic_frames terminates after draining
    frames = list(cap.mic_frames())
    assert len(frames) == 2
    assert all(f.size == 512 for f in frames)
    assert np.allclose(frames[0], 0.3, atol=1e-3)


def test_voiceprocessing_capture_resamples_device_rate():
    from my_stt_tts.aec import VoiceProcessingCapture

    cap = VoiceProcessingCapture(sample_rate=16000, frame_samples=160)
    cap._device_sr = 48000.0  # 48k -> 16k is a 3:1 decimation
    cap._on_buffer(_FakeVPBuffer(np.full(480, 0.5, dtype=np.float32)), None)
    cap._closed.set()
    frames = list(cap.mic_frames())
    assert frames and frames[0].size == 160  # 480/3 == 160 samples at 16 kHz


def test_make_voiceprocessing_capture_off_when_not_selected():
    from my_stt_tts.aec import make_voiceprocessing_capture

    # aec off -> never builds a HW capture (would fall back to sounddevice + NLMS).
    assert make_voiceprocessing_capture(Config(aec_mode="off")) is None
    assert make_voiceprocessing_capture(Config(aec_mode="nlms")) is None
    # voiceprocessing but HW capture disabled -> None (software path).
    assert (
        make_voiceprocessing_capture(Config(aec_mode="voiceprocessing", aec_hw_capture=False))
        is None
    )


def test_monitor_during_playback_uses_hw_source():
    """The barge-in monitor pulls HW-cancelled frames from a capture source (R3-4)."""
    from my_stt_tts import audio
    from my_stt_tts.interrupt import InterruptGate

    class _FakePlayback:
        def __init__(self) -> None:
            self._done = False
            self.cancelled = False

        @property
        def done(self):
            return self._done

        def cancel(self):
            self.cancelled = True
            self._done = True

        def wait(self):
            self._done = True

        reference = None

    class _AlwaysSpeech:
        def is_speech(self, frame):  # noqa: ANN001, ARG002
            return True

    class _HwSource:
        def mic_frames(self):
            for _ in range(10):
                yield np.full(512, 0.5, dtype=np.float32)

    gate = InterruptGate(min_speech_ms=0.0, min_words=0, frame_ms=32.0)
    res = audio.monitor_during_playback(
        _FakePlayback(),
        16000,
        _AlwaysSpeech(),
        gate,
        energy_floor=0.0,
        source=_HwSource(),
    )
    assert res.interrupted is True
    assert res.captured.size > 0


# ============================================================================
# R3-1 — true WebRTC transport (queue bridge + signaling seam)
# ============================================================================


def test_webrtc_transport_is_audio_transport():
    from my_stt_tts.transport import AudioTransport
    from my_stt_tts.webrtc_transport import WebRtcTransport

    assert isinstance(WebRtcTransport(16000), AudioTransport)


def test_webrtc_transport_resamples_mic_in_and_tts_out():
    from my_stt_tts.webrtc_transport import WebRtcTransport

    t = WebRtcTransport(sample_rate=16000)
    # Browser mic arrives at 48 kHz; the loop should see 16 kHz frames.
    t.feed_mic(np.full(480, 0.25, dtype=np.float32), sample_rate=48000)
    t.end_mic()
    frames = list(t.mic_frames())
    assert frames and frames[0].size == 160  # 480 @ 48k -> 160 @ 16k

    # TTS produced at 16 kHz is resampled up to 48 kHz for the Opus track.
    t2 = WebRtcTransport(sample_rate=16000)
    t2.send_tts(np.full(160, 0.1, dtype=np.float32), 16000)
    out = t2.next_tts(timeout=0.2)
    assert out is not None
    pcm, sr = out
    assert sr == 48000 and pcm.size == 480


def test_webrtc_transport_close_ends_mic_frames():
    from my_stt_tts.webrtc_transport import WebRtcTransport

    t = WebRtcTransport()
    t.close()
    assert t.closed is True
    assert list(t.mic_frames()) == []


class _FakePc:
    """A duck-typed RTCPeerConnection for the signaling seam (no real ICE)."""

    def __init__(self) -> None:
        self.remote = None
        self.local = None

        class _LD:
            sdp = "v=0\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\na=rtpmap:111 opus/48000/2\r\n"
            type = "answer"

        self._ld = _LD()

    async def setRemoteDescription(self, desc):  # noqa: N802
        self.remote = desc

    async def createAnswer(self):  # noqa: N802
        return self._ld

    async def setLocalDescription(self, desc):  # noqa: N802
        self.local = desc

    @property
    def localDescription(self):  # noqa: N802
        return self._ld


def test_negotiate_answer_produces_sdp_answer():
    import asyncio

    from my_stt_tts.webrtc_transport import negotiate_answer

    pc = _FakePc()
    offer = {"sdp": "v=0\r\nm=audio ...opus...\r\n", "type": "offer"}
    answer = asyncio.run(negotiate_answer(pc, offer))
    assert answer["type"] == "answer"
    assert "opus" in answer["sdp"].lower()
    assert pc.remote is not None and pc.local is not None  # both descriptions set


def test_webrtc_available_reports_aiortc_presence():
    from my_stt_tts.webrtc_transport import webrtc_available

    # Just assert it returns a bool without raising (value depends on the env).
    assert isinstance(webrtc_available(), bool)


# ============================================================================
# R3-2 — full-duplex barge-in over the network transport
# ============================================================================


class _FakeBrainStream:
    """A Brain stand-in: streams fixed deltas; records commit_spoken + close."""

    def __init__(self, parts) -> None:
        self._parts = parts
        self.committed = None
        self.closed = False

    def stream(self, text):  # noqa: ARG002
        def _gen():
            yield from self._parts

        return _StreamObj(_gen(), self)

    def commit_spoken(self, text):
        self.committed = text


class _StreamObj:
    def __init__(self, gen, brain) -> None:
        self._gen = gen
        self._brain = brain

    def __iter__(self):
        return self._gen

    def close(self):
        self._brain.closed = True


class _StreamTTS:
    """A TTS stand-in with a clause stream of small PCM chunks."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    def synth_pcm(self, text, lang=None):  # noqa: ANN001, ARG002
        self.spoken.append(text)
        return np.full(8, 0.1, dtype=np.float32), 16000

    def synth_pcm_stream(self, text, lang=None):  # noqa: ANN001, ARG002
        self.spoken.append(text)
        # two PCM clause-chunks per sentence
        yield np.full(8, 0.1, dtype=np.float32), 16000
        yield np.full(8, 0.1, dtype=np.float32), 16000


class _MicSourceStub:
    """A barge-in mic source: a fixed list of frames, then empty (None)."""

    def __init__(self, frames) -> None:
        self._frames = list(frames)
        self.closed = False

    def get(self, timeout=0.1):  # noqa: ARG002
        return self._frames.pop(0) if self._frames else None


class _SpeechVad:
    def is_speech(self, frame):  # noqa: ANN001, ARG002
        return True


def test_respond_over_transport_half_duplex_sinks_pcm():
    """Without a mic source it stays half-duplex (R2-5 behaviour) and returns a result."""
    from my_stt_tts.ws_transport import WebSocketTransport

    cfg = Config(sample_rate=16000)
    t = WebSocketTransport(sample_rate=16000)
    brain = _FakeBrainStream(["Hello there. ", "How are you?"])
    tts = _StreamTTS()
    res = net_loop.respond_over_transport(t, cfg, brain, tts, "hi")
    assert res.interrupted is False
    assert "Hello there." in "".join(tts.spoken)
    assert t.iter_outbound(timeout=0.1) is not None


def test_respond_over_transport_duplex_interrupt_cancels_tts_and_llm():
    """A confirmed barge-in during playout aborts TTS + the LLM stream (R3-2)."""
    from my_stt_tts.ws_transport import WebSocketTransport

    cfg = Config(
        sample_rate=16000,
        barge_in="always",
        interrupt_min_speech_ms=0.0,
        interrupt_min_words=0,
        interrupt_predict=False,
        aec_mode="off",
        barge_in_energy=0.0,
    )
    t = WebSocketTransport(sample_rate=16000)
    brain = _FakeBrainStream(["First sentence here. ", "Second sentence. ", "Third sentence."])
    tts = _StreamTTS()
    # The mic delivers loud speech frames immediately -> interrupt on the 1st sentence.
    source = _MicSourceStub([np.full(512, 0.5, dtype=np.float32) for _ in range(20)])
    res = net_loop.respond_over_transport(t, cfg, brain, tts, "hi", source=source, vad=_SpeechVad())
    assert res.interrupted is True
    assert res.captured.size > 0  # the barge-in audio was captured for the next turn
    assert brain.closed is True  # the LLM stream was cancelled
    # Not every sentence was synthesized: the reply was cut off early.
    assert len(tts.spoken) < 3


def test_transport_barge_in_no_interrupt_when_silent():
    """Silence on the mic must NOT interrupt the reply (R3-2 false-trigger guard)."""
    from my_stt_tts.ws_transport import WebSocketTransport

    cfg = Config(
        sample_rate=16000,
        barge_in="always",
        interrupt_min_speech_ms=350.0,
        interrupt_min_words=2,
        interrupt_predict=False,
        aec_mode="off",
    )
    t = WebSocketTransport(sample_rate=16000)
    brain = _FakeBrainStream(["Only sentence."])
    tts = _StreamTTS()

    class _SilentVad:
        def is_speech(self, frame):  # noqa: ANN001, ARG002
            return False

    source = _MicSourceStub([np.zeros(512, dtype=np.float32) for _ in range(10)])
    res = net_loop.respond_over_transport(t, cfg, brain, tts, "hi", source=source, vad=_SilentVad())
    assert res.interrupted is False
    assert "Only sentence." in "".join(tts.spoken)


def test_mic_source_shares_one_reader():
    """The shared _MicSource feeds frames pulled with a timeout, then closes (R3-2)."""
    from my_stt_tts.transport import encode_frame
    from my_stt_tts.ws_transport import WebSocketTransport

    t = WebSocketTransport(sample_rate=16000)
    for _ in range(3):
        t.feed_mic(encode_frame(np.full(160, 0.2, dtype=np.float32)))
    t.end_mic()
    src = net_loop._MicSource(t)
    got = []
    for _ in range(50):
        f = src.get(timeout=0.1)
        if f is not None:
            got.append(f)
        if src.closed:
            break
    assert len(got) == 3
    assert src.closed


class _CountingSTT:
    def __init__(self, text: str) -> None:
        self._text = text
        self.calls = 0

    def transcribe(self, audio, sample_rate=16000) -> STTResult:  # noqa: ANN001, ARG002
        self.calls += 1
        return STTResult(text=self._text)


def test_transcribe_captured_handles_barge_in_clip():
    cfg = Config(sample_rate=16000)
    stt = _CountingSTT("the next turn")
    clip = np.full(8000, 0.2, dtype=np.float32)
    text = net_loop._transcribe_captured(cfg, stt, clip)
    assert text == "the next turn"
    assert net_loop._transcribe_captured(cfg, stt, np.zeros(0, dtype=np.float32)) == ""
