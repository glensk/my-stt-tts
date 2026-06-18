"""Tests for the natural-conversation features (Phase 7):

* G1 — cancellable TTS playback (abort signalling)
* G2 — TurnAnalyzer protocol + fallback selection
* G4 — false-interrupt suppression gate
* G5 — post-interruption context repair (history truncation)
* G6 — streaming STT chunk boundaries

The audio/subprocess boundaries are faked so nothing needs a real mic or GPU.
"""
# pylint: disable=missing-function-docstring,protected-access

from unittest.mock import patch

import numpy as np

from my_stt_tts import audio
from my_stt_tts.brain import Brain
from my_stt_tts.config import Config
from my_stt_tts.interrupt import InterruptGate, frame_energy, word_count
from my_stt_tts.stt import StreamingTranscriber, STTResult, stream_transcribe
from my_stt_tts.tts import Playback
from my_stt_tts.turn import (
    SilenceTurnAnalyzer,
    SmartTurnAnalyzer,
    TurnAnalyzer,
    make_turn_analyzer,
)

# --- G4: false-interrupt suppression ------------------------------------------


def test_word_count_ignores_punctuation():
    assert word_count("Hey, what's up?") == 3
    assert word_count("") == 0
    assert word_count("mhm") == 1


def test_frame_energy_silence_vs_loud():
    assert frame_energy(np.zeros(512, dtype=np.float32)) == 0.0
    loud = frame_energy(np.full(512, 0.5, dtype=np.float32))
    assert abs(loud - 0.5) < 1e-5


def test_gate_opens_after_min_speech_ms():
    # 100 ms frames; need 350 ms => opens on the 4th voiced frame.
    gate = InterruptGate(min_speech_ms=350.0, min_words=99, frame_ms=100.0)
    assert gate.update(True) is False  # 100
    assert gate.update(True) is False  # 200
    assert gate.update(True) is False  # 300
    assert gate.update(True) is True  # 400 >= 350
    assert gate.open is True


def test_gate_ignores_backchannel_blip():
    # A single 100 ms "mhm" with min 350 ms and 5 words required => never opens.
    gate = InterruptGate(min_speech_ms=350.0, min_words=5, frame_ms=100.0)
    assert gate.update(True, partial_text="mhm") is False
    assert gate.update(False) is False
    assert gate.open is False


def test_gate_opens_on_word_count_before_duration():
    gate = InterruptGate(min_speech_ms=5000.0, min_words=2, frame_ms=100.0)
    assert gate.update(True, partial_text="stop") is False  # 1 word
    assert gate.update(True, partial_text="stop now") is True  # 2 words


def test_gate_requires_some_speech_even_with_words():
    # No voiced frame yet => stays closed regardless of text.
    gate = InterruptGate(min_speech_ms=0.0, min_words=1, frame_ms=100.0)
    assert gate.update(False, partial_text="hello there") is False


def test_gate_both_guards_disabled_any_speech_interrupts():
    gate = InterruptGate(min_speech_ms=0.0, min_words=0, frame_ms=100.0)
    assert gate.update(True) is True


def test_gate_reset_clears_state():
    gate = InterruptGate(min_speech_ms=100.0, min_words=0, frame_ms=100.0)
    assert gate.update(True) is True
    gate.reset()
    assert gate.open is False
    assert gate.voiced_ms == 0.0


# --- G5: post-interruption context repair -------------------------------------


def _streamed_brain(reply_parts: list[str]) -> Brain:
    """A Brain whose backend yields ``reply_parts`` (no network)."""
    brain = Brain(Config(llm_provider="anthropic", anthropic_api_key="x"))
    brain._stream_anthropic = lambda model: iter(reply_parts)  # type: ignore[assignment]
    return brain


def test_commit_spoken_truncates_to_voiced_prefix():
    brain = _streamed_brain(["Hello there. ", "This part was never spoken."])
    full = "".join(brain.stream("hi"))
    assert full == "Hello there. This part was never spoken."
    # Only "Hello there." was actually voiced before the barge-in.
    brain.commit_spoken("Hello there.")
    assert brain.history[-1] == {"role": "assistant", "content": "Hello there."}


def test_commit_spoken_empty_drops_assistant_turn():
    brain = _streamed_brain(["Some reply that was cut off immediately."])
    list(brain.stream("hi"))
    assert brain.history[-1]["role"] == "assistant"
    brain.commit_spoken("")  # nothing was voiced
    assert brain.history[-1] == {"role": "user", "content": "hi"}


def test_commit_spoken_noop_without_pending():
    brain = _streamed_brain(["a reply"])
    list(brain.stream("hi"))
    brain.commit_spoken("a reply")  # consumes the pending index
    before = list(brain.history)
    brain.commit_spoken("ignored")  # second call is a no-op
    assert brain.history == before


def test_full_reply_kept_when_not_interrupted():
    brain = _streamed_brain(["The whole answer stays."])
    list(brain.stream("hi"))
    # No commit_spoken call => full generated reply remains in history.
    assert brain.history[-1]["content"] == "The whole answer stays."


# --- G2: TurnAnalyzer protocol + fallback selection ---------------------------


def test_silence_analyzer_satisfies_protocol():
    analyzer = SilenceTurnAnalyzer(0.3, frame_seconds=0.1)
    assert isinstance(analyzer, TurnAnalyzer)


def test_silence_analyzer_ends_after_silence():
    analyzer = SilenceTurnAnalyzer(0.3, frame_seconds=0.1)
    frame = np.zeros(512, dtype=np.float32)
    assert analyzer.update(frame, True) is False
    assert analyzer.update(frame, False) is False
    assert analyzer.update(frame, False) is False
    assert analyzer.update(frame, False) is True


def test_make_turn_analyzer_defaults_to_silence():
    cfg = Config(turn_analyzer="silence")
    assert isinstance(make_turn_analyzer(cfg, 0.032), SilenceTurnAnalyzer)


def test_make_turn_analyzer_smart_selects_smart():
    cfg = Config(turn_analyzer="smart", smart_turn_model_path="/no/such/model.onnx")
    analyzer = make_turn_analyzer(cfg, 0.032)
    assert isinstance(analyzer, SmartTurnAnalyzer)


def test_smart_turn_falls_back_to_silence_when_model_missing():
    # Model file absent => behaves exactly like the silence endpointer (short
    # silence ends the turn) without raising.
    analyzer = SmartTurnAnalyzer("/no/such/model.onnx", silence_seconds=0.2, frame_seconds=0.1)
    frame = np.zeros(512, dtype=np.float32)
    voiced = np.full(512, 0.2, dtype=np.float32)
    assert analyzer.update(voiced, True) is False
    assert analyzer.update(frame, False) is False
    assert analyzer.update(frame, False) is True  # 0.2s silence -> end (fallback)
    assert analyzer._fallback is True


def test_smart_turn_consults_model_on_candidate():
    # Fake the loaded model: return "not complete" once, then "complete".
    analyzer = SmartTurnAnalyzer(
        "/fake.onnx", silence_seconds=0.2, frame_seconds=0.1, threshold=0.5
    )
    probs = iter([0.1, 0.9])
    analyzer._ensure_model = lambda: True  # type: ignore[assignment]
    analyzer._completion_probability = lambda: next(probs)  # type: ignore[assignment]
    voiced = np.full(512, 0.2, dtype=np.float32)
    silence = np.zeros(512, dtype=np.float32)
    # speak, then a short pause -> candidate -> model says 0.1 (keep going)
    assert analyzer.update(voiced, True) is False
    assert analyzer.update(silence, False) is False
    assert analyzer.update(silence, False) is False  # prob 0.1 -> not done
    # speak again, pause again -> model says 0.9 -> end
    assert analyzer.update(voiced, True) is False
    assert analyzer.update(silence, False) is False
    assert analyzer.update(silence, False) is True  # prob 0.9 -> done


# --- G6: streaming STT chunk boundaries ---------------------------------------


class _FakeSTT:
    """Returns text proportional to how many samples it has seen."""

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> STTResult:
        self.calls += 1
        words = max(1, audio.shape[0] // sample_rate)  # ~1 word/sec
        return STTResult(text=" ".join(["word"] * words))


def test_streamer_partials_at_interval_boundaries():
    engine = _FakeSTT()
    # 1000-sample frames, 16 kHz, partial every 600 ms (9600 samples).
    streamer = StreamingTranscriber(engine, sample_rate=16000, partial_interval_ms=600.0)
    frame = np.ones(1000, dtype=np.float32)
    partials = [streamer.feed(frame) for _ in range(20)]
    produced = [p for p in partials if p is not None]
    # 20 frames = 20000 samples; partial fires at >= 9600 samples since last one,
    # so roughly every ~10 frames => about 2 partials.
    assert len(produced) >= 1
    # First partial only after at least the interval of audio accumulated.
    first_idx = next(i for i, p in enumerate(partials) if p is not None)
    assert first_idx >= 9  # 10 * 1000 >= 9600


def test_streamer_skips_duplicate_partials():
    class _ConstSTT:
        def transcribe(self, audio, sample_rate=16000):  # noqa: ANN001, ARG002
            return STTResult(text="same")

    streamer = StreamingTranscriber(_ConstSTT(), sample_rate=1000, partial_interval_ms=100.0)
    frame = np.ones(200, dtype=np.float32)
    out = [streamer.feed(frame) for _ in range(5)]
    produced = [p for p in out if p is not None]
    assert produced == ["same"]  # text unchanged -> only emitted once


def test_streamer_final_transcribes_full_buffer():
    engine = _FakeSTT()
    streamer = StreamingTranscriber(engine, sample_rate=16000, partial_interval_ms=600.0)
    for _ in range(3):
        streamer.feed(np.ones(16000, dtype=np.float32))  # 3 seconds total
    final = streamer.final()
    assert final.text.split() == ["word", "word", "word"]


def test_streamer_reset_clears_buffer():
    engine = _FakeSTT()
    streamer = StreamingTranscriber(engine, sample_rate=16000)
    streamer.feed(np.ones(16000, dtype=np.float32))
    streamer.reset()
    assert streamer.final().text == ""


def test_stream_transcribe_emits_partials_and_returns_final():
    engine = _FakeSTT()
    frames = iter([np.ones(16000, dtype=np.float32) for _ in range(3)])
    seen: list[str] = []
    final = stream_transcribe(
        engine, frames, sample_rate=16000, partial_interval_ms=600.0, on_partial=seen.append
    )
    assert seen  # at least one partial fired
    assert final.text.split() == ["word", "word", "word"]


# --- G1: cancellable playback abort signalling --------------------------------


class _FakeProc:
    """A subprocess.Popen stand-in: starts 'running', kill() makes poll() return."""

    def __init__(self) -> None:
        self._returncode: int | None = None
        self.killed = False

    def poll(self):
        return self._returncode

    def kill(self) -> None:
        self.killed = True
        self._returncode = -9

    def wait(self):
        self._returncode = self._returncode if self._returncode is not None else 0
        return self._returncode


def test_playback_cancel_kills_process_and_sets_flag():
    proc = _FakeProc()
    pb = Playback(proc)
    assert pb.cancelled is False
    assert pb.done is False
    pb.cancel()
    assert proc.killed is True
    assert pb.cancelled is True
    assert pb.done is True


def test_playback_cancel_is_idempotent_after_exit():
    proc = _FakeProc()
    pb = Playback(proc)
    proc.wait()  # process finished on its own
    assert pb.done is True
    pb.cancel()  # must not raise even though it's already done
    assert pb.cancelled is True


def test_empty_playback_is_inert():
    pb = Playback()
    assert pb.done is True
    pb.cancel()  # no subprocess -> no-op
    pb.wait()
    assert pb.cancelled is True


def test_playback_wait_returns_after_completion():
    proc = _FakeProc()
    pb = Playback(proc)
    pb.wait()
    assert proc.poll() == 0


# --- G1: barge-in monitor (mic live during playback) --------------------------


class _FakePlayback:
    """Reports 'playing' for ``polls_until_done`` checks of ``done`` then finishes.

    ``monitor_during_playback`` checks ``done`` once per loop iteration, so this
    bounds the loop for the no-interrupt case without needing a real subprocess.
    """

    def __init__(self, polls_until_done: int = 1000) -> None:
        self._left = polls_until_done
        self.cancelled = False

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
    """Context manager that feeds pre-seeded frames to the capture callback."""

    def __init__(self, frames: list[np.ndarray], callback) -> None:  # noqa: ANN001
        self._frames = frames
        self._callback = callback

    def __enter__(self):
        for frame in self._frames:
            self._callback(frame.reshape(-1, 1), len(frame), None, None)
        return self

    def __exit__(self, *_exc) -> None:
        return None


def _fake_sd(frames: list[np.ndarray]):
    class _SD:
        @staticmethod
        def InputStream(*, callback, **_kw):  # noqa: N802, ANN003
            return _FakeStream(frames, callback)

    return _SD()


class _AlwaysSpeechVad:
    def is_speech(self, frame) -> bool:  # noqa: ANN001, ARG002
        return True


def test_monitor_interrupts_on_confirmed_speech_and_captures_audio():
    voiced = [np.full(512, 0.3, dtype=np.float32) for _ in range(6)]
    gate = InterruptGate(min_speech_ms=64.0, min_words=0, frame_ms=32.0)
    playback = _FakePlayback()
    with patch.object(audio, "_sd", lambda: _fake_sd(voiced)):
        result = audio.monitor_during_playback(
            playback, 16000, _AlwaysSpeechVad(), gate, energy_floor=0.02, poll_seconds=0.01
        )
    assert result.interrupted is True
    assert playback.cancelled is True
    assert result.captured.size > 0  # the barge-in speech was kept for the next turn


def test_monitor_ignores_low_energy_bleed():
    # Frames below the energy floor (speaker bleed) -> never a barge-in.
    bleed = [np.full(512, 0.005, dtype=np.float32) for _ in range(6)]
    gate = InterruptGate(min_speech_ms=32.0, min_words=0, frame_ms=32.0)
    playback = _FakePlayback(polls_until_done=20)
    with patch.object(audio, "_sd", lambda: _fake_sd(bleed)):
        result = audio.monitor_during_playback(
            playback, 16000, _AlwaysSpeechVad(), gate, energy_floor=0.02, poll_seconds=0.01
        )
    assert result.interrupted is False
    assert playback.cancelled is False
