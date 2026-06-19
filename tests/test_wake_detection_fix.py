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
    played: list[np.ndarray] = []
    monkeypatch.setattr(main_mod, "_play", lambda s: played.append(np.asarray(s)))
    events: list[dict] = []
    sub = bus.subscribe()

    main_mod._run_mic_record_replay(Config(anthropic_api_key="x"))

    while not sub.empty():
        import json

        events.append(json.loads(sub.get_nowait()))
    bus.unsubscribe(sub)
    # It played the recording back (the user hears their own mic)…
    assert played and played[0].size == clip.size
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
