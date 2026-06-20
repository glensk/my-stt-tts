"""Wake-detection fix + related backend features (wake-detection-fix branch).

Covers, with the audio/model boundaries mocked (the REAL maziko.onnx verification
is a manual check — see PLAN.md / the report):

* the int16 scale conversion fed to openWakeWord (the never-fires root cause):
  ``WakeWord.detect`` must hand ``predict`` **int16** samples, not float32 [-1, 1]
  (0.4.0's AudioFeatures truncates a float signal to zeros);
* :func:`to_int16_pcm` scaling / int16 pass-through;
* the contiguous-1280-frame streaming buffer in :func:`listen_for_wake` (no
  zero-padding mid-stream; leftover samples buffered across blocks);
* a distinct wake chime exists and is played on detection, with ``bus.wake`` fired;
* source tagging threaded onto ``bus.transcript``;
* the ``mic_record_replay`` action handler + the mic-confirmed signal.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring,too-many-arguments,too-many-positional-arguments

from __future__ import annotations

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from my_stt_tts.wake import WakeWord, to_int16_pcm


# --------------------------------------------------------------------------- #
# (1) THE FIX: detect() feeds int16 PCM, not float32 — the never-fires bug     #
# --------------------------------------------------------------------------- #
def _install_fake_openwakeword(monkeypatch: pytest.MonkeyPatch, model_cls: type) -> None:
    pkg = types.ModuleType("openwakeword")
    mod = types.ModuleType("openwakeword.model")
    mod.Model = model_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.model", mod)


def test_detect_feeds_int16_pcm_to_predict(monkeypatch: pytest.MonkeyPatch) -> None:
    """The core regression guard: a float32 [-1, 1] capture frame must reach
    ``predict`` as int16 PCM (±32768). openWakeWord 0.4.0 truncates float input to
    zeros, so feeding float is exactly why the score was pinned at ~0.001."""
    seen: dict[str, Any] = {}

    class CaptureModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, frame: np.ndarray) -> dict[str, float]:
            seen["dtype"] = frame.dtype
            seen["peak"] = int(np.max(np.abs(frame)))
            return {"maziko": 0.0}

    _install_fake_openwakeword(monkeypatch, CaptureModel)
    w = WakeWord("wakewords/maziko.onnx")
    # A realistic capture frame: float32 in [-1, 1] (what the mic/resample deliver).
    frame = (np.sin(np.linspace(0, 50, 1280)) * 0.4).astype(np.float32)
    w.detect(frame)
    assert seen["dtype"] == np.int16  # NOT float32 — this is the whole bug
    assert seen["peak"] > 1000  # 0.4 full-scale -> ~13000, NOT truncated to 0


def test_float_frame_does_not_truncate_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mimic 0.4.0's AudioFeatures: float input would be cast int16→all zeros. With
    the fix the model receives real, non-zero PCM, so it can actually score."""
    captured: dict[str, int] = {}

    class TruncatingModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, frame: np.ndarray) -> dict[str, float]:
            # Replicate openWakeWord's list→int16 buffering on whatever we were fed.
            as_int16 = np.array(list(frame)).astype(np.int16)
            captured["nonzero"] = int(np.count_nonzero(as_int16))
            return {"maziko": 0.0}

    _install_fake_openwakeword(monkeypatch, TruncatingModel)
    w = WakeWord("wakewords/maziko.onnx")
    frame = (np.sin(np.linspace(0, 50, 1280)) * 0.4).astype(np.float32)
    w.detect(frame)
    assert captured["nonzero"] > 100  # NOT 0 — the audio survives the int16 round-trip


def test_to_int16_pcm_scales_float_and_passes_int16_through() -> None:
    f = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    out = to_int16_pcm(f)
    assert out.dtype == np.int16
    assert out[0] == -32767 and out[-1] == 32767 and out[2] == 0
    # Out-of-range floats are clipped, not wrapped.
    assert to_int16_pcm(np.array([2.0, -3.0], dtype=np.float32)).tolist() == [32767, -32767]
    # An already-int16 frame is returned unchanged (no double scaling).
    i16 = np.array([1, -2, 30000], dtype=np.int16)
    assert to_int16_pcm(i16) is i16


def test_detect_fires_when_int16_audio_scores_high(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end through detect(): given a model that scores high on real int16
    energy, a float capture frame now fires (it would never have, fed as float)."""

    class EnergyModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, frame: np.ndarray) -> dict[str, float]:
            # "Fires" only if it actually received non-trivial int16 PCM.
            score = 0.9 if int(np.max(np.abs(frame))) > 1000 else 0.001
            return {"maziko": score}

    _install_fake_openwakeword(monkeypatch, EnergyModel)
    w = WakeWord("wakewords/maziko.onnx", threshold=0.5)
    loud = (np.sin(np.linspace(0, 50, 1280)) * 0.4).astype(np.float32)
    assert w.detect(loud) is True
    assert w.last_score == pytest.approx(0.9)
    quiet = np.zeros(1280, dtype=np.float32)
    assert w.detect(quiet) is False


# --------------------------------------------------------------------------- #
# (2) Contiguous 1280-frame streaming buffer in listen_for_wake               #
# --------------------------------------------------------------------------- #
class _FakeStream:
    """Drives ``_callback`` with a scripted list of device blocks, then idles. The
    test's fake wake detector sets the ``stop`` event once it has seen the expected
    frames, so the real ``while True`` loop drains the queue then exits cleanly."""

    def __init__(self, blocks: list[np.ndarray], callback) -> None:
        self._blocks = blocks
        self._cb = callback

    def __enter__(self) -> _FakeStream:
        for block in self._blocks:
            self._cb(block.reshape(-1, 1), len(block), None, None)
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_listen_for_wake_feeds_contiguous_1280_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """The model must see EXACT 1280-sample frames assembled contiguously across
    device blocks — never a zero-padded short tail mid-stream (which would inject
    silence gaps and corrupt openWakeWord's stateful feature accumulation)."""
    import threading

    from my_stt_tts import audio

    # Device blocks of odd sizes (1000 + 800 + 760 = 2560 = exactly 2 frames).
    blocks = [
        np.full(1000, 0.1, dtype=np.float32),
        np.full(800, 0.2, dtype=np.float32),
        np.full(760, 0.3, dtype=np.float32),
    ]
    fed: list[np.ndarray] = []
    stop = threading.Event()

    class RecordingWake:
        threshold = 0.5
        model_name = "maziko"
        last_score = 0.0

        def reset(self) -> None:
            return None

        def detect(self, frame: np.ndarray) -> bool:
            fed.append(np.asarray(frame).copy())
            if len(fed) >= 2:  # both full frames seen -> let the loop exit (no fire)
                stop.set()
            return False  # never fire; we only inspect the framing

    sd = MagicMock()
    sd.InputStream.side_effect = lambda **kw: _FakeStream(blocks, kw["callback"])
    monkeypatch.setattr(audio, "_sd", lambda: sd)
    monkeypatch.setattr(audio, "_supported_capture_rate", lambda _sd, r: r)

    audio.listen_for_wake(RecordingWake(), 16000, poll_seconds=0.01, stop=stop)
    assert len(fed) == 2  # 2560 samples -> exactly two 1280 frames, no padded 3rd
    for frame in fed:
        assert frame.shape == (1280,)
    # Frames are contiguous: every sample is a real captured value, no injected zeros.
    assert not np.any(fed[0] == 0.0) and not np.any(fed[1] == 0.0)
    # Leftover < 1280 is buffered across blocks, never zero-padded into a frame.
    joined = np.concatenate(fed)
    assert joined[0] == pytest.approx(0.1)  # first block's value leads


def test_listen_for_wake_returns_true_on_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    from my_stt_tts import audio

    blocks = [np.full(1280, 0.5, dtype=np.float32)]

    class FiringWake:
        threshold = 0.5
        model_name = "maziko"
        last_score = 0.99

        def reset(self) -> None:
            return None

        def detect(self, _frame: np.ndarray) -> bool:
            return True

    sd = MagicMock()
    sd.InputStream.side_effect = lambda **kw: _FakeStream(blocks, kw["callback"])
    monkeypatch.setattr(audio, "_sd", lambda: sd)
    monkeypatch.setattr(audio, "_supported_capture_rate", lambda _sd, r: r)
    assert audio.listen_for_wake(FiringWake(), 16000, poll_seconds=0.01) is True


# --------------------------------------------------------------------------- #
# (3) source tagging on bus.transcript                                        #
# --------------------------------------------------------------------------- #
def test_transcript_source_tag_in_payload() -> None:
    from my_stt_tts.events import EventBus

    bus = EventBus()
    sub = bus.subscribe()
    bus.transcript("hello", source="push_to_talk")
    import json

    payload = json.loads(sub.get_nowait())
    assert payload["type"] == "transcript"
    assert payload["source"] == "push_to_talk"


def test_transcript_source_omitted_for_backcompat() -> None:
    from my_stt_tts.events import EventBus

    bus = EventBus()
    sub = bus.subscribe()
    bus.transcript("hi")  # no source -> field absent (back-compat)
    import json

    payload = json.loads(sub.get_nowait())
    assert "source" not in payload


# --------------------------------------------------------------------------- #
# (4) wake beep/cue: a distinct chime exists and detection fires bus.wake      #
# --------------------------------------------------------------------------- #
def test_chime_wake_is_distinct_nonempty() -> None:
    from my_stt_tts import chimes

    wake = chimes.chime_wake()
    listening = chimes.chime_listening()
    assert wake.dtype == np.float32 and wake.size > 0
    # A different cue from the mic-live chime (different length / content).
    assert wake.size != listening.size or not np.array_equal(wake, listening)


# --------------------------------------------------------------------------- #
# (5) mic_record_replay action handler + the mic-confirmed (hide-hint) signal #
# --------------------------------------------------------------------------- #
class _FakeUI:
    last_on_action = None

    def __init__(self, _cfg, _on_turn, on_action=None, **_kwargs) -> None:
        type(self).last_on_action = on_action

    def url(self) -> str:
        return "http://127.0.0.1:8765/"

    def serve_forever(self) -> None:
        return None


def test_mic_record_replay_action_runs_without_controller() -> None:
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.config import Config

    cfg = Config(anthropic_api_key="x")
    with (
        patch("my_stt_tts.webui.WebUI", _FakeUI),
        patch("webbrowser.open"),
        patch.object(main_mod, "_run_mic_record_replay") as run_replay,
    ):
        main_mod._run_browser(cfg, MagicMock(), MagicMock(), MagicMock(), None, wake=False)
        handler = _FakeUI.last_on_action
        assert callable(handler)
        handler("mic_record_replay", {})  # pylint: disable=not-callable
    import time

    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not run_replay.called:
        time.sleep(0.01)
    run_replay.assert_called_once()


def test_run_mic_record_replay_records_plays_and_reports(monkeypatch: pytest.MonkeyPatch) -> None:
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.config import Config
    from my_stt_tts.events import bus

    clip = (np.sin(np.linspace(0, 30, 16000)) * 0.5).astype(np.float32)
    monkeypatch.setattr(main_mod.audio, "record_fixed", lambda _sr, seconds=3.0: (clip, 48000))
    monkeypatch.setattr(main_mod.audio, "mic_permission_status", lambda *_a: "authorized")
    played: list[tuple[np.ndarray, int]] = []
    # Replay goes through audio.play(clip, device_rate) so it plays at the CAPTURE
    # rate (faithful pitch) — not _play (which hardcodes the 24 kHz chime rate).
    monkeypatch.setattr(
        main_mod.audio, "play", lambda s, sr, *_a, **_k: played.append((np.asarray(s), sr))
    )
    events: list[dict] = []
    sub = bus.subscribe()

    main_mod._run_mic_record_replay(Config(anthropic_api_key="x"))

    while not sub.empty():
        import json

        events.append(json.loads(sub.get_nowait()))
    bus.unsubscribe(sub)
    # It played the recording back (the user hears their own mic)…
    assert played and played[0][0].size == clip.size
    # …at the SAME rate it was captured at (48 kHz device rate, not 16/24 kHz) so
    # the round-trip is faithful — this is the speed/pitch fix.
    assert played[0][1] == 48000
    # …and emitted an ok mic_result (the hide-permission-hint signal).
    results = [e for e in events if e.get("type") == "mic_result"]
    assert results and results[-1]["ok"] is True
    assert results[-1]["level"] > 0


def test_run_mic_record_replay_silent_capture_reports_not_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.config import Config
    from my_stt_tts.events import bus

    silent = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(main_mod.audio, "record_fixed", lambda _sr, seconds=3.0: (silent, 48000))
    monkeypatch.setattr(main_mod.audio, "mic_permission_status", lambda *_a: "authorized")
    monkeypatch.setattr(main_mod, "_play", lambda _s: None)
    sub = bus.subscribe()
    main_mod._run_mic_record_replay(Config(anthropic_api_key="x"))
    import json

    results = []
    while not sub.empty():
        ev = json.loads(sub.get_nowait())
        if ev.get("type") == "mic_result":
            results.append(ev)
    bus.unsubscribe(sub)
    assert results and results[-1]["ok"] is False  # silent -> not ok


def test_signal_mic_confirmed_emits_ok_on_real_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.events import bus

    monkeypatch.setattr(main_mod.audio, "mic_permission_status", lambda *_a: "authorized")
    sub = bus.subscribe()
    main_mod._signal_mic_confirmed(
        (np.sin(np.linspace(0, 30, 8000)) * 0.5).astype(np.float32), 16000
    )
    import json

    results = [json.loads(sub.get_nowait()) for _ in range(sub.qsize()) if not sub.empty()]
    bus.unsubscribe(sub)
    oks = [e for e in results if e.get("type") == "mic_result" and e.get("ok")]
    assert oks, "a confirmed capture must emit mic_result(ok=True) to hide the perm hint"


def test_signal_mic_confirmed_noop_on_silence() -> None:
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.events import bus

    sub = bus.subscribe()
    # Drain the replayed last-state frame the bus sends to new subscribers.
    while not sub.empty():
        sub.get_nowait()
    main_mod._signal_mic_confirmed(np.zeros(8000, dtype=np.float32), 16000)
    assert sub.empty()  # silence proves nothing -> no signal
    bus.unsubscribe(sub)


# --------------------------------------------------------------------------- #
# (6) Phase-diverse detection: the "fires offline, never live" recall fix     #
# --------------------------------------------------------------------------- #
class _PhaseModel:
    """Fake openWakeWord model that scores high ONLY when a frame captures the
    ENTIRE wake "spike" (every sample non-zero) — i.e. only when the 1280-frame
    grid lands exactly on the word. This models openWakeWord's real phase
    sensitivity: a frame straddling the word's edge (half zeros) scores ~0. Only a
    detector whose grid phase aligns with the word sees a clean, all-on frame."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        return {"maziko": 0.9 if np.all(frame != 0) else 0.0}

    def reset(self) -> None:
        return None


def test_phases_default_one_preserves_single_detector(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare WakeWord(...) defaults to ONE phase model (back-compat)."""
    _install_fake_openwakeword(monkeypatch, _PhaseModel)
    w = WakeWord("wakewords/maziko.onnx")
    w._ensure()
    assert w.phases == 1
    assert len(w._models) == 1


def test_phase_diversity_fires_when_single_phase_misses(monkeypatch: pytest.MonkeyPatch) -> None:
    """With phases>1, a wake word that lands on an unlucky frame phase for the
    canonical grid still fires via one of the staggered detectors — the live-recall
    fix. The signal is offset by half a frame so phase-0's grid straddles it."""
    _install_fake_openwakeword(monkeypatch, _PhaseModel)
    # A 1280-sample wake "spike" placed at offset 640 (half a frame). The phases=1
    # grid only ever sees frames [0:1280] and [1280:2560] — each straddles the spike
    # edge (half zeros) -> 0.0, so it NEVER fires. A detector offset by +640 samples
    # sees frame [640:1920] = the full spike -> fires. Trailing zeros flush the grids.
    stream = np.concatenate(
        [
            np.zeros(640, dtype=np.float32),
            np.full(1280, 0.5, dtype=np.float32),
            np.zeros(1280, dtype=np.float32),
        ]
    )

    def run(phases: int) -> bool:
        w = WakeWord("wakewords/maziko.onnx", threshold=0.5, phases=phases)
        w.reset()
        pending, fired = stream.copy(), False
        while pending.size >= 1280:
            frame, pending = pending[:1280], pending[1280:]
            if w.detect(frame):
                fired = True
        return fired

    assert run(1) is False  # single phase straddles the spike edge -> never fires
    assert run(8) is True  # a staggered phase lands cleanly on the spike -> fires


def test_phase_models_all_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """reset() must clear every phase model's state AND re-prime the offset buffers."""
    reset_calls = {"n": 0}

    class CountingReset(_PhaseModel):
        def reset(self) -> None:
            reset_calls["n"] += 1

    _install_fake_openwakeword(monkeypatch, CountingReset)
    w = WakeWord("wakewords/maziko.onnx", phases=4)
    w._ensure()
    w.reset()
    assert reset_calls["n"] == 4  # one per phase model
    # Buffers re-primed with staggered lead zeros: 0, 320, 640, 960 samples.
    assert [b.size for b in w._pending] == [0, 320, 640, 960]


# --------------------------------------------------------------------------- #
# (7) Wake-debug recorder: dumps the EXACT 16 kHz frames + logs stats/score    #
# --------------------------------------------------------------------------- #
def test_wake_debug_recorder_writes_wav_and_logs(tmp_path) -> None:  # noqa: ANN001
    """The recorder taps the first N s of 16 kHz frames, writes a mono 16 kHz WAV,
    and logs sample rate / #samples / duration / rms / peak / max+mean score."""
    import wave

    from my_stt_tts.audio import WakeDebugRecorder

    events: list[tuple[str, dict[str, Any]]] = []

    def on_debug(stage: str, **fields: Any) -> None:
        events.append((stage, fields))

    path = tmp_path / "wake-debug.wav"
    # 0.16 s window at 16 kHz = 2560 samples = exactly two 1280 frames.
    rec = WakeDebugRecorder(str(path), 16000, 0.16, on_debug=on_debug)
    frame = (np.sin(np.linspace(0, 20, 1280)) * 0.5).astype(np.float32)
    rec.feed(frame, 0.2)
    assert not rec.done  # one frame is below the window -> still filling
    rec.feed(frame, 0.6)
    assert rec.done  # window full -> flushed once
    rec.feed(frame, 0.9)  # post-flush feed is a no-op (no second write)

    # A valid 16 kHz mono 16-bit WAV exists.
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getnframes() == 2560

    # A single wake_debug event carrying the right telemetry.
    wake_dbg = [f for s, f in events if s == "wake_debug"]
    assert len(wake_dbg) == 1
    f = wake_dbg[0]
    assert f["sample_rate"] == 16000
    assert f["samples"] == 2560
    assert f["duration_s"] == pytest.approx(0.16, abs=1e-3)
    assert f["peak"] > 0.4  # 0.5 full-scale sine
    assert f["level_pct"] == 50
    assert f["max_score"] == 0.6  # NOT the post-flush 0.9 (window closed)
    assert f["mean_score"] == pytest.approx(0.4, abs=1e-3)  # mean(0.2, 0.6)
    assert str(path) in str(f["path"])


def test_wake_debug_recorder_silent_capture_reports_low_level(tmp_path) -> None:  # noqa: ANN001
    """A near-silent capture is flagged by level_pct≈0 — the capture-problem signal
    (wrong rate / no mic permission) vs a model-recall problem (good audio, low score)."""
    from my_stt_tts.audio import WakeDebugRecorder

    events: list[dict[str, Any]] = []
    rec = WakeDebugRecorder(
        str(tmp_path / "silent.wav"),
        16000,
        0.08,
        on_debug=lambda _s, **f: events.append(f),
    )
    rec.feed(np.zeros(1280, dtype=np.float32), 0.001)
    assert rec.done
    assert events[0]["level_pct"] == 0  # near-silent -> capture problem, not recall
    assert events[0]["max_score"] == 0.001


def test_listen_for_wake_feeds_exact_frames_to_recorder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The recorder gets the SAME post-resample frames the model scores — not a
    separate capture (proves the WAV is the real wake-model input)."""
    import threading

    from my_stt_tts import audio
    from my_stt_tts.audio import WakeDebugRecorder

    # Two contiguous 1280 frames worth of a known ramp at the device (=pipeline) rate.
    blocks = [np.linspace(0.1, 0.9, 2560, dtype=np.float32)]
    fed_to_model: list[np.ndarray] = []
    stop = threading.Event()

    class RecordingWake:
        threshold = 0.5
        model_name = "maziko"
        last_score = 0.0

        def reset(self) -> None:
            return None

        def detect(self, frame: np.ndarray) -> bool:
            fed_to_model.append(np.asarray(frame).copy())
            if len(fed_to_model) >= 2:
                stop.set()
            return False

    captured: list[np.ndarray] = []

    class TapRecorder(WakeDebugRecorder):
        def feed(self, frame: np.ndarray, score: float) -> None:  # noqa: D102
            captured.append(np.asarray(frame).copy())

    sd = MagicMock()
    sd.InputStream.side_effect = lambda **kw: _FakeStream(blocks, kw["callback"])
    monkeypatch.setattr(audio, "_sd", lambda: sd)
    monkeypatch.setattr(audio, "_supported_capture_rate", lambda _sd, r: r)

    rec = TapRecorder("/tmp/_unused.wav", 16000, 5.0)
    audio.listen_for_wake(RecordingWake(), 16000, poll_seconds=0.01, stop=stop, recorder=rec)

    # The recorder saw exactly the frames the model saw, in order.
    assert len(captured) == len(fed_to_model) == 2
    for cap, fed in zip(captured, fed_to_model, strict=True):
        assert np.array_equal(cap, fed)
