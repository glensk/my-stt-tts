"""Tests for the round-2 natural-conversation gaps:

* R2-1 — acoustic echo cancellation (the EchoCanceller seam + NLMS adaptive filter
  + the monitor-loop wiring that relaxes the energy floor when AEC is active)
* R2-2 — bounded sliding-window streaming STT (window stitching + bounded re-decode)
* R2-3 — acoustic interruption prediction (intent scoring on synthetic features)
* R2-4 — smart-turn by default + the model auto-download guard (network mocked)
* R2-6 — robust interrupt plumbing (bus events + the captured-audio hand-off)

All audio / subprocess / network boundaries are faked so nothing needs a real
mic, GPU, or network.
"""
# pylint: disable=missing-function-docstring,protected-access,missing-class-docstring

from unittest.mock import patch

import numpy as np

from my_stt_tts import audio
from my_stt_tts.aec import (
    AEC_MODES,
    EchoCanceller,
    NlmsEchoCanceller,
    NullEchoCanceller,
    VoiceProcessingEchoCanceller,
    make_echo_canceller,
)
from my_stt_tts.config import Config
from my_stt_tts.events import bus
from my_stt_tts.interrupt import (
    InterruptGate,
    InterruptPredictor,
    make_interrupt_predictor,
)
from my_stt_tts.stt import StreamingTranscriber, STTResult, stitch_partial
from my_stt_tts.turn import SmartTurnAnalyzer, ensure_smart_turn_model, make_turn_analyzer

# --- R2-1: acoustic echo cancellation -----------------------------------------


def test_null_canceller_is_identity_and_inactive():
    aec = NullEchoCanceller()
    assert aec.active is False
    frame = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    aec.push_reference(np.ones(10, dtype=np.float32))  # ignored
    assert np.allclose(aec.process(frame), frame)


def test_null_and_nlms_satisfy_echo_canceller_protocol():
    assert isinstance(NullEchoCanceller(), EchoCanceller)
    assert isinstance(NlmsEchoCanceller(taps=8), EchoCanceller)


def test_nlms_passes_through_until_reference_available():
    aec = NlmsEchoCanceller(taps=16, mu=0.5)
    mic = np.full(8, 0.25, dtype=np.float32)
    # Constructor seeds `taps` zeros, so a short frame with no real reference yet
    # has a full (zero) window => predicts ~0 echo => returns the mic unchanged.
    out = aec.process(mic)
    assert out.shape == mic.shape
    assert np.allclose(out, mic, atol=1e-6)


def test_nlms_cancels_a_linear_echo_path():
    # The mic hears a delayed, scaled copy of the loudspeaker reference (the echo)
    # plus nothing else; the adaptive filter should drive the residual down.
    rng = np.random.default_rng(0)
    sr = 16000
    ref = (rng.standard_normal(sr) * 0.3).astype(np.float32)
    delay = 20
    echo = np.zeros_like(ref)
    echo[delay:] = 0.6 * ref[:-delay]
    aec = NlmsEchoCanceller(taps=64, mu=0.5)
    aec.push_reference(ref)
    cleaned = aec.process(echo)
    # After the filter has converged (skip the initial transient), residual echo
    # energy is far below the raw echo: real cancellation, not a no-op.
    raw_rms = float(np.sqrt(np.mean(echo[2000:] ** 2)))
    res_rms = float(np.sqrt(np.mean(cleaned[2000:] ** 2)))
    assert res_rms < raw_rms * 0.5  # >= ~6 dB ERLE
    erle_db = 20.0 * np.log10(raw_rms / max(res_rms, 1e-9))
    assert erle_db > 6.0


def test_nlms_preserves_user_speech_under_echo():
    # Echo + a user burst: the user energy must survive cancellation.
    rng = np.random.default_rng(1)
    sr = 16000
    ref = (rng.standard_normal(sr) * 0.3).astype(np.float32)
    delay = 20
    echo = np.zeros_like(ref)
    echo[delay:] = 0.6 * ref[:-delay]
    user = np.zeros_like(ref)
    user[8000:8400] = 0.4  # the user's interruption
    aec = NlmsEchoCanceller(taps=64, mu=0.5)
    aec.push_reference(ref)
    cleaned = aec.process(echo + user)
    user_band = float(np.sqrt(np.mean(cleaned[8000:8400] ** 2)))
    quiet_band = float(np.sqrt(np.mean(cleaned[4000:4400] ** 2)))
    assert user_band > quiet_band * 3  # user speech stands out from cancelled echo


def test_nlms_reset_clears_filter():
    aec = NlmsEchoCanceller(taps=8, mu=0.5)
    aec.push_reference(np.ones(64, dtype=np.float32))
    aec.process(np.ones(8, dtype=np.float32))
    aec.reset()
    assert np.count_nonzero(aec._w) == 0
    assert len(aec._ref) == aec.taps  # re-seeded with `taps` zeros


def test_nlms_rejects_bad_params():
    for kwargs in ({"taps": 0}, {"mu": 0.0}, {"mu": 3.0}):
        try:
            NlmsEchoCanceller(**kwargs)  # type: ignore[arg-type]
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs}")


def test_make_echo_canceller_modes():
    assert isinstance(make_echo_canceller(Config(aec_mode="off")), NullEchoCanceller)
    assert isinstance(make_echo_canceller(Config(aec_mode="nlms")), NlmsEchoCanceller)
    # Force "no hardware" so auto/voiceprocessing both fall back to NLMS even on a
    # Mac where the API exists.
    with patch.object(VoiceProcessingEchoCanceller, "available", staticmethod(lambda: False)):
        assert isinstance(make_echo_canceller(Config(aec_mode="auto")), NlmsEchoCanceller)
        assert isinstance(
            make_echo_canceller(Config(aec_mode="voiceprocessing")), NlmsEchoCanceller
        )


def test_voiceprocessing_falls_back_to_nlms_when_inactive():
    # Hardware "available" but the unit refuses to enable -> NLMS, not a dead HW AEC.
    class _Inactive(VoiceProcessingEchoCanceller):
        def __init__(self):  # noqa: D107
            self.active = False

    with (
        patch.object(VoiceProcessingEchoCanceller, "available", staticmethod(lambda: True)),
        patch("my_stt_tts.aec.VoiceProcessingEchoCanceller", _Inactive),
    ):
        assert isinstance(make_echo_canceller(Config(aec_mode="auto")), NlmsEchoCanceller)


def test_aec_modes_constant_matches_config():
    # G8 added the Linux WebRTC-APM AEC backend ("webrtc").
    assert AEC_MODES == ("off", "nlms", "voiceprocessing", "webrtc", "auto")


# --- R2-1: monitor-loop wiring (AEC processes frames + relaxes the floor) ------


class _FakePlayback:
    def __init__(self, polls_until_done=1000, reference=None, reference_sr=None):  # noqa: ANN001
        self._left = polls_until_done
        self.cancelled = False
        self.reference = reference
        self.reference_sr = reference_sr

    @property
    def done(self) -> bool:
        if self.cancelled:
            return True
        self._left -= 1
        return self._left < 0

    def cancel(self) -> None:
        self.cancelled = True

    def wait(self) -> None:
        self._left = 0


class _FakeStream:
    def __init__(self, frames, callback) -> None:  # noqa: ANN001
        self._frames = frames
        self._callback = callback

    def __enter__(self):
        for frame in self._frames:
            self._callback(frame.reshape(-1, 1), len(frame), None, None)
        return self

    def __exit__(self, *_exc) -> None:
        return None


def _fake_sd(frames):
    class _SD:
        @staticmethod
        def InputStream(*, callback, **_kw):  # noqa: N802, ANN003
            return _FakeStream(frames, callback)

    return _SD()


class _AlwaysSpeechVad:
    def is_speech(self, frame) -> bool:  # noqa: ANN001, ARG002
        return True


class _RecordingAec:
    """An AEC stub that records that it was fed the reference + each frame."""

    active = True

    def __init__(self) -> None:
        self.reference_pushed: list[int] = []
        self.processed = 0

    def push_reference(self, samples) -> None:  # noqa: ANN001
        self.reference_pushed.append(np.asarray(samples).size)

    def process(self, frame):  # noqa: ANN001
        self.processed += 1
        return np.asarray(frame, dtype=np.float32).ravel()

    def reset(self) -> None:
        self.reference_pushed.clear()
        self.processed = 0


def test_monitor_pushes_reference_and_processes_each_frame():
    voiced = [np.full(512, 0.3, dtype=np.float32) for _ in range(4)]
    gate = InterruptGate(min_speech_ms=10000.0, min_words=0, frame_ms=32.0)  # never via gate
    aec = _RecordingAec()
    ref = np.full(8000, 0.1, dtype=np.float32)
    playback = _FakePlayback(polls_until_done=4, reference=ref, reference_sr=16000)
    with patch.object(audio, "_sd", lambda: _fake_sd(voiced)):
        audio.monitor_during_playback(
            playback, 16000, _AlwaysSpeechVad(), gate, energy_floor=0.02, aec=aec
        )
    assert aec.reference_pushed and aec.reference_pushed[0] == ref.size
    assert aec.processed == len(voiced)


def test_monitor_relaxes_energy_floor_when_aec_active():
    # Quiet speech (below the open-speaker bleed floor) WOULD be ignored without
    # AEC; with an active AEC the floor is relaxed so it can interrupt.
    quiet = [np.full(512, 0.006, dtype=np.float32) for _ in range(6)]
    gate = InterruptGate(min_speech_ms=32.0, min_words=0, frame_ms=32.0)
    aec = _RecordingAec()  # active=True, identity process
    playback = _FakePlayback(polls_until_done=20)
    with patch.object(audio, "_sd", lambda: _fake_sd(quiet)):
        result = audio.monitor_during_playback(
            playback, 16000, _AlwaysSpeechVad(), gate, energy_floor=0.02, aec=aec
        )
    assert result.interrupted is True  # not suppressed as bleed


def test_monitor_predictor_can_interrupt_before_gate():
    # The gate's word + duration guards are set absurdly high, so ONLY the acoustic
    # predictor can authorise the interruption (R2-3 composed into the loop).
    rng = np.random.default_rng(3)
    voiced = [(rng.standard_normal(512) * 0.2).astype(np.float32) for _ in range(15)]
    gate = InterruptGate(min_speech_ms=1e9, min_words=999, frame_ms=32.0)
    predictor = InterruptPredictor(threshold=0.6, frame_ms=32.0, min_ms=200.0, energy_floor=0.02)
    playback = _FakePlayback(polls_until_done=30)
    with patch.object(audio, "_sd", lambda: _fake_sd(voiced)):
        result = audio.monitor_during_playback(
            playback, 16000, _AlwaysSpeechVad(), gate, energy_floor=0.02, predictor=predictor
        )
    assert result.interrupted is True
    assert gate.open is False  # the gate never fired; the predictor did


# --- R2-2: bounded sliding-window streaming STT --------------------------------


def test_stitch_partial_dedupes_word_overlap():
    assert stitch_partial("what is the", "is the weather") == "what is the weather"
    assert stitch_partial("a b c", "c d e") == "a b c d e"
    assert stitch_partial("", "hello world") == "hello world"
    assert stitch_partial("good morning", "") == "good morning"
    # No overlap -> plain concatenation.
    assert stitch_partial("one two", "three four") == "one two three four"


class _SizeSpySTT:
    """Records every decoded clip length; returns ~1 word per second of audio."""

    def __init__(self) -> None:
        self.sizes: list[int] = []

    def transcribe(self, audio_in: np.ndarray, sample_rate: int = 16000) -> STTResult:
        self.sizes.append(audio_in.shape[0])
        words = max(1, audio_in.shape[0] // sample_rate)
        return STTResult(text=" ".join(["w"] * words))


def test_streaming_window_bounds_redecode_size():
    sr = 16000
    spy = _SizeSpySTT()
    st = StreamingTranscriber(spy, sample_rate=sr, partial_interval_ms=1000.0, window_s=4.0)
    for _ in range(30):  # 30 s utterance in 1 s frames
        st.feed(np.ones(sr, dtype=np.float32))
    window_n = 4 * sr
    # Every single decode is bounded (<= ~1.5 windows), NOT the growing 30 s buffer.
    assert max(spy.sizes) <= int(1.5 * window_n) + sr
    # And it really is bounded vs the total audio that was fed.
    assert max(spy.sizes) < 10 * sr


def test_streaming_window_final_uses_full_buffer():
    sr = 16000
    spy = _SizeSpySTT()
    st = StreamingTranscriber(spy, sample_rate=sr, partial_interval_ms=1000.0, window_s=4.0)
    for _ in range(12):
        st.feed(np.ones(sr, dtype=np.float32))
    final = st.final()
    assert len(final.text.split()) == 12  # full 12 s decoded for accuracy
    assert spy.sizes[-1] == 12 * sr  # the final pass saw the whole buffer


def test_streaming_short_utterance_decodes_whole_clip():
    # Below the window, behaviour matches re-decoding the whole buffer (no stitch).
    sr = 16000
    spy = _SizeSpySTT()
    st = StreamingTranscriber(spy, sample_rate=sr, partial_interval_ms=1000.0, window_s=8.0)
    for _ in range(3):
        st.feed(np.ones(sr, dtype=np.float32))
    assert all(s <= 3 * sr for s in spy.sizes)


def test_streaming_partial_stitches_committed_prefix():
    # A CONTENT-aware fake engine: each 10-sample block carries a distinct integer
    # marker, decoded to a distinct token. Windows at different positions therefore
    # yield different tokens, so a correct stitch accumulates a prefix longer than
    # any single window decode (proving committed text is folded in, not dropped).
    sr = 100

    class _ContentSTT:
        def transcribe(self, audio_in: np.ndarray, sample_rate: int = sr) -> STTResult:
            tokens = [f"t{int(round(float(audio_in[i])))}" for i in range(0, audio_in.size, 10)]
            return STTResult(text=" ".join(tokens))

    st = StreamingTranscriber(
        _ContentSTT(), sample_rate=sr, partial_interval_ms=100.0, window_s=1.0
    )
    partial = None
    marker = 0
    for _ in range(40):  # 4 s of audio, window only 1 s
        block = np.full(50, 0.0, dtype=np.float32)
        # Each 10-sample sub-block gets a unique increasing marker value.
        for j in range(0, 50, 10):
            block[j : j + 10] = marker
            marker += 1
        out = st.feed(block)
        if out is not None:
            partial = out
    assert partial is not None
    # A single 1 s window decode is only ~10 tokens; stitching the committed prefix
    # makes the partial substantially longer (accumulated history).
    assert len(partial.split()) > 12
    # And the tokens are monotonically increasing markers (correct ordering, no dups).
    nums = [int(tok[1:]) for tok in partial.split()]
    assert nums == sorted(nums)
    assert len(nums) == len(set(nums))


# --- R2-3: acoustic interruption prediction ------------------------------------


def _noise(rng, scale=0.2, n=512):  # noqa: ANN001
    return (rng.standard_normal(n) * scale).astype(np.float32)


def test_predictor_ignores_short_backchannel():
    rng = np.random.default_rng(2)
    p = InterruptPredictor(threshold=0.6, frame_ms=32.0, min_ms=240.0, energy_floor=0.02)
    fired = False
    for _ in range(2):  # a 64 ms "mhm"
        fired = p.update(_noise(rng), True) or fired
    for _ in range(10):  # then silence
        fired = p.update(np.zeros(512, dtype=np.float32), False) or fired
    assert fired is False
    assert p.open is False


def test_predictor_fires_on_sustained_speech():
    rng = np.random.default_rng(2)
    p = InterruptPredictor(threshold=0.6, frame_ms=32.0, min_ms=240.0, energy_floor=0.02)
    first = None
    for i in range(20):
        if p.update(_noise(rng), True) and first is None:
            first = i
    assert p.open is True
    assert first is not None
    # Must respect the duration floor (240 ms / 32 ms ~ 8 frames) before firing.
    assert first >= 240.0 / 32.0 - 1


def test_predictor_requires_duration_even_if_loud():
    # A single very loud frame must not fire (transient/cough rejection).
    p = InterruptPredictor(threshold=0.0, frame_ms=32.0, min_ms=240.0, energy_floor=0.02)
    assert p.update(np.full(512, 0.9, dtype=np.float32), True) is False


def test_predictor_score_decays_on_silence():
    rng = np.random.default_rng(4)
    p = InterruptPredictor(threshold=0.95, frame_ms=32.0, min_ms=32.0, energy_floor=0.02)
    for _ in range(3):
        p.update(_noise(rng), True)
    peak = p.score
    for _ in range(5):
        p.update(np.zeros(512, dtype=np.float32), False)
    assert p.score < peak


def test_predictor_reset_clears_state():
    rng = np.random.default_rng(5)
    p = InterruptPredictor(threshold=0.1, frame_ms=32.0, min_ms=0.0, energy_floor=0.02)
    p.update(_noise(rng), True)
    p.reset()
    assert p.score == 0.0
    assert p.open is False


def test_make_interrupt_predictor_respects_config_flag():
    assert make_interrupt_predictor(Config(interrupt_predict=False), 32.0) is None
    pred = make_interrupt_predictor(
        Config(interrupt_predict=True, interrupt_predict_threshold=0.7), 32.0
    )
    assert isinstance(pred, InterruptPredictor)
    assert pred.threshold == 0.7


# --- R2-4: smart-turn by default + auto-download guard -------------------------


def test_turn_analyzer_defaults_to_smart():
    assert Config().turn_analyzer == "smart"
    assert isinstance(make_turn_analyzer(Config(), 0.032), SmartTurnAnalyzer)


def test_smart_turn_default_is_passed_download_settings():
    cfg = Config(
        turn_analyzer="smart",
        smart_turn_model_url="https://example.test/model.onnx",
        smart_turn_auto_download=True,
    )
    analyzer = make_turn_analyzer(cfg, 0.032)
    assert isinstance(analyzer, SmartTurnAnalyzer)
    assert analyzer.model_url == "https://example.test/model.onnx"
    assert analyzer.auto_download is True


def test_ensure_smart_turn_model_present_skips_download(tmp_path):
    model = tmp_path / "smart-turn.onnx"
    model.write_bytes(b"already here")
    # urlopen must NOT be called when the file exists.
    with patch("my_stt_tts.turn.urllib.request.urlopen", side_effect=AssertionError("no net")):
        assert ensure_smart_turn_model(str(model), "https://x/y.onnx") is True


def test_ensure_smart_turn_model_downloads_when_missing(tmp_path):
    model = tmp_path / "models" / "smart-turn.onnx"

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        @staticmethod
        def read() -> bytes:
            return b"ONNX-BYTES"

    with patch("my_stt_tts.turn.urllib.request.urlopen", return_value=_Resp()) as mocked:
        ok = ensure_smart_turn_model(str(model), "https://example.test/m.onnx")
    assert ok is True
    assert model.read_bytes() == b"ONNX-BYTES"
    mocked.assert_called_once()


def test_ensure_smart_turn_model_no_download_when_disabled(tmp_path):
    model = tmp_path / "smart-turn.onnx"
    with patch("my_stt_tts.turn.urllib.request.urlopen", side_effect=AssertionError("no net")):
        assert ensure_smart_turn_model(str(model), "https://x/y.onnx", auto_download=False) is False
    assert not model.exists()


def test_ensure_smart_turn_model_network_failure_falls_back(tmp_path):
    model = tmp_path / "smart-turn.onnx"
    import urllib.error

    with patch(
        "my_stt_tts.turn.urllib.request.urlopen",
        side_effect=urllib.error.URLError("offline"),
    ):
        assert ensure_smart_turn_model(str(model), "https://x/y.onnx") is False
    assert not model.exists()  # no half-written file left behind


def test_smart_turn_analyzer_auto_downloads_on_first_use(tmp_path):
    model = tmp_path / "smart-turn.onnx"
    analyzer = SmartTurnAnalyzer(
        str(model),
        silence_seconds=0.2,
        frame_seconds=0.1,
        model_url="https://example.test/m.onnx",
        auto_download=True,
    )
    voiced = np.full(512, 0.2, dtype=np.float32)
    silence = np.zeros(512, dtype=np.float32)
    # The model "downloads" but onnxruntime/transformers aren't installed, so the
    # analyzer still gracefully falls back to silence — and tried the download.
    with patch("my_stt_tts.turn.ensure_smart_turn_model", return_value=False) as ensure_mock:
        analyzer.update(voiced, True)
        analyzer.update(silence, False)
        ended = analyzer.update(silence, False)
    assert ended is True  # silence fallback ended the turn
    assert analyzer._fallback is True
    ensure_mock.assert_called()


# --- R2-6: robust interrupt plumbing (bus events + audio hand-off) -------------


class _BusSpy:
    """Subscribes to the bus and collects published event dicts."""

    def __init__(self) -> None:
        import json

        self._json = json
        self._sub = bus.subscribe()

    def drain(self) -> list[dict]:
        out = []
        while True:
            try:
                out.append(self._json.loads(self._sub.get_nowait()))
            except Exception:  # queue.Empty
                break
        return out

    def close(self) -> None:
        bus.unsubscribe(self._sub)


def test_bus_emits_interrupt_lifecycle_events():
    spy = _BusSpy()
    try:
        spy.drain()  # clear the replayed last-state
        bus.interrupt_start()
        bus.bot_stopped_speaking()
        bus.interrupt_stop()
        events = spy.drain()
    finally:
        spy.close()
    types = [(e.get("type"), e.get("phase")) for e in events]
    assert ("interrupt", "start") in types
    assert ("bot_stopped_speaking", None) in types
    assert ("interrupt", "stop") in types


def test_feed_clip_hands_off_captured_audio_without_rescratch():
    sr = 16000

    class _CountSTT:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_in: np.ndarray, sample_rate: int = sr) -> STTResult:
            self.calls += 1
            return STTResult(text="captured words here")

    engine = _CountSTT()
    st = StreamingTranscriber(engine, sample_rate=sr, partial_interval_ms=600.0)
    captured = np.ones(sr, dtype=np.float32)  # 1 s of barge-in audio
    st.feed_clip(captured)  # hand-off: must NOT decode yet
    assert engine.calls == 0
    # The next live frame extends the SAME buffer (no from-scratch reset).
    st.feed(np.ones(sr, dtype=np.float32))
    final = st.final()
    assert final.text == "captured words here"
    assert engine.calls >= 1


def test_transcribe_barge_in_uses_streamer_and_emits_stop():
    from my_stt_tts import __main__ as cli

    sr = 16000
    cfg = Config(stt_streaming=True, stt_window_s=7.0)

    class _CountSTT:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, audio_in: np.ndarray, sample_rate: int = sr) -> STTResult:
            self.calls += 1
            return STTResult(text="next turn please")

    stt = _CountSTT()
    spy = _BusSpy()
    try:
        spy.drain()
        text = cli._transcribe_barge_in(cfg, stt, np.ones(sr, dtype=np.float32))
        events = spy.drain()
    finally:
        spy.close()
    assert text == "next turn please"
    # Hand-off via the streamer => exactly ONE final decode (no extra re-transcribe).
    assert stt.calls == 1
    assert ("interrupt", "stop") in [(e.get("type"), e.get("phase")) for e in events]


def test_transcribe_barge_in_empty_clip_still_emits_stop():
    from my_stt_tts import __main__ as cli

    spy = _BusSpy()
    try:
        spy.drain()
        text = cli._transcribe_barge_in(Config(), object(), np.zeros(0, dtype=np.float32))
        events = spy.drain()
    finally:
        spy.close()
    assert text == ""
    assert ("interrupt", "stop") in [(e.get("type"), e.get("phase")) for e in events]
