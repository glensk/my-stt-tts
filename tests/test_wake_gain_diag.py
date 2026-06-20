"""Wake-gain diagnostics: the live-loop gain fix + the GUI sweep / score / stats.

The PRIME suspect for the dead wake word: openWakeWord has NO input gain
normalization, so a quiet mic -> low mel energies -> the score collapses (~0.001)
regardless of the word. This suite covers the centerpiece fix + diagnostics:

* ``wake_gain`` config field (from_env / validate / settings round-trip) and the
  clip-protected gain applied BEFORE ``WakeWord.detect`` in the LIVE wake loop
  (``audio.listen_for_wake``).
* ``score_wake_clip(..., gain=, with_trace=)`` — the gain param and the per-frame
  ``score_trace`` shape.
* The ``wake_gain_sweep`` action: confidence rises with gain on a quiet synthetic
  clip (the smoking gun) — and the ``wake_gain_sweep_result`` / ``score_clip_result``
  emitters.
* ``score_clip`` action: scores the EXACT saved WAV (capture-vs-model isolation).
* ``audio.capture_stats`` extensions: int16_peak/int16_rms, expected_level
  classification, crest_db / dc_offset / true_peak_db / snr_db math, and lufs gated
  on the optional ``pyloudnorm`` dep.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring,import-outside-toplevel

from __future__ import annotations

import importlib.util
import json
import queue
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from my_stt_tts import __main__ as main_mod
from my_stt_tts import audio
from my_stt_tts.config import Config, ConfigError
from my_stt_tts.events import EventBus
from my_stt_tts.wake import WakeWord, score_wake_clip


# --------------------------------------------------------------------------- #
# Fake openWakeWord plumbing                                                  #
# --------------------------------------------------------------------------- #
def _install_fake_openwakeword(monkeypatch: pytest.MonkeyPatch, model_cls: type) -> None:
    pkg = types.ModuleType("openwakeword")
    mod = types.ModuleType("openwakeword.model")
    mod.Model = model_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.model", mod)


class _LevelModel:
    """Scores monotonically with the frame's int16 peak — a stand-in for the real
    model's gain sensitivity: louder input (more energy) -> higher score, capped at 1.

    This is exactly the behaviour the gain stage exploits: a quiet clip scores low,
    and the SAME clip amplified scores higher.
    """

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        peak = float(np.max(np.abs(frame))) if frame.size else 0.0
        # int16 frames from to_int16_pcm range to ~32767; map peak -> [0, 1).
        return {"maziko": min(0.999, peak / 32768.0)}


def _drain(sub: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while True:
        try:
            out.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    return out


def _write_wav(path: Path, clip: np.ndarray, rate: int = 16000) -> None:
    from my_stt_tts.util import wav_bytes_from_float

    path.write_bytes(wav_bytes_from_float(np.asarray(clip, dtype=np.float32).ravel(), rate))


# --------------------------------------------------------------------------- #
# 1. wake_gain config field                                                   #
# --------------------------------------------------------------------------- #
def test_wake_gain_defaults_to_one() -> None:
    assert Config().wake_gain == 1.0


def test_wake_gain_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAKE_GAIN", "4.0")
    cfg = Config.from_env()
    assert cfg.wake_gain == 4.0


def test_wake_gain_validate_rejects_out_of_range() -> None:
    too_high = Config(anthropic_api_key="sk-test", wake_gain=11.0)
    with pytest.raises(ConfigError, match="wake_gain"):
        too_high.validate()
    zero = Config(anthropic_api_key="sk-test", wake_gain=0.0)
    with pytest.raises(ConfigError, match="wake_gain"):
        zero.validate()


def test_wake_gain_validate_accepts_in_range() -> None:
    Config(anthropic_api_key="sk-test", wake_gain=2.5).validate()  # no raise


def test_wake_gain_settings_round_trip() -> None:
    from my_stt_tts.webui import apply_settings, settings_dict

    cfg = Config(anthropic_api_key="sk-test")
    assert settings_dict(cfg)["wake_gain"] == 1.0
    apply_settings(cfg, {"wake_gain": 3.0})
    assert cfg.wake_gain == 3.0
    # A hand-crafted out-of-range POST is clamped to validate()'s bound.
    apply_settings(cfg, {"wake_gain": 50.0})
    assert cfg.wake_gain == 10.0


def test_wake_gain_in_settings_text() -> None:
    cfg = Config(anthropic_api_key="sk-test", wake_gain=2.0)
    assert "wake-gain" in main_mod.settings_text(cfg, color=False)


# --------------------------------------------------------------------------- #
# 2. live-loop gain: applied (clip-protected) BEFORE WakeWord.detect          #
# --------------------------------------------------------------------------- #
class _FakeStream:
    """A one-block sounddevice InputStream stand-in: feeds the callback once."""

    def __init__(self, block: np.ndarray, **_kwargs: Any) -> None:
        self._block = block
        self._cb = _kwargs.get("callback")

    def __enter__(self) -> _FakeStream:
        if self._cb is not None:
            self._cb(self._block.reshape(-1, 1), len(self._block), None, None)
        return self

    def __exit__(self, *_exc: Any) -> None:
        pass


def _fake_sd(block: np.ndarray) -> Any:
    sd = types.SimpleNamespace()
    sd.InputStream = lambda **kw: _FakeStream(block, **kw)
    sd.check_input_settings = lambda **_k: None
    sd.query_devices = lambda **_k: {"default_samplerate": 16000.0}
    return sd


def test_listen_for_wake_applies_gain_before_detect(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live loop multiplies each frame by ``gain`` (clip-protected) before detect."""
    block = np.full(1280, 0.1, dtype=np.float32)  # quiet block
    monkeypatch.setattr(audio, "_sd", lambda: _fake_sd(block))

    seen_peaks: list[float] = []

    class _Wake:
        threshold = 0.5
        last_score = 0.0
        model_name = "maziko"

        def reset(self) -> None:
            pass

        def detect(self, frame: np.ndarray) -> bool:
            seen_peaks.append(float(np.max(np.abs(frame))))
            self.last_score = 0.9  # fire immediately so the loop returns
            return True

    fired = audio.listen_for_wake(_Wake(), 16000, gain=4.0)
    assert fired is True
    # 0.1 * 4 = 0.4 reaches detect (NOT the raw 0.1) -> the gain was applied first.
    assert seen_peaks and seen_peaks[0] == pytest.approx(0.4, abs=1e-4)


def test_listen_for_wake_gain_clip_protected(monkeypatch: pytest.MonkeyPatch) -> None:
    """A hot frame * gain is hard-clipped to ±1.0 (never wraps)."""
    block = np.full(1280, 0.5, dtype=np.float32)
    monkeypatch.setattr(audio, "_sd", lambda: _fake_sd(block))
    seen_peaks: list[float] = []

    class _Wake:
        threshold = 0.5
        last_score = 0.0
        model_name = "maziko"

        def reset(self) -> None:
            pass

        def detect(self, frame: np.ndarray) -> bool:
            seen_peaks.append(float(np.max(np.abs(frame))))
            self.last_score = 0.9
            return True

    audio.listen_for_wake(_Wake(), 16000, gain=8.0)  # 0.5 * 8 = 4.0 -> clipped to 1.0
    assert seen_peaks and seen_peaks[0] == pytest.approx(1.0)


def test_listen_for_wake_gain_one_is_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    block = np.full(1280, 0.1, dtype=np.float32)
    monkeypatch.setattr(audio, "_sd", lambda: _fake_sd(block))
    seen_peaks: list[float] = []

    class _Wake:
        threshold = 0.5
        last_score = 0.0
        model_name = "maziko"

        def reset(self) -> None:
            pass

        def detect(self, frame: np.ndarray) -> bool:
            seen_peaks.append(float(np.max(np.abs(frame))))
            self.last_score = 0.9
            return True

    audio.listen_for_wake(_Wake(), 16000, gain=1.0)
    assert seen_peaks and seen_peaks[0] == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# 3. score_wake_clip gain + score_trace                                       #
# --------------------------------------------------------------------------- #
def test_score_wake_clip_gain_lifts_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A quiet clip scores low at gain 1; the SAME clip scores higher at higher gain."""
    _install_fake_openwakeword(monkeypatch, _LevelModel)
    quiet = np.full(32000, 0.02, dtype=np.float32)  # ~655 int16 peak
    low, _ = score_wake_clip(quiet, 16000, "maziko", threshold=0.5, phases=1, gain=1.0)
    high, _ = score_wake_clip(quiet, 16000, "maziko", threshold=0.5, phases=1, gain=8.0)
    assert high > low
    assert low < 0.1  # near-silent to the model at unity gain


def test_score_wake_clip_with_trace_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """with_trace returns (conf, fired, trace) where trace has one score per frame."""
    _install_fake_openwakeword(monkeypatch, _LevelModel)
    clip = np.full(16000, 0.3, dtype=np.float32)  # 1 s -> ~12 frames of 1280
    result = score_wake_clip(clip, 16000, "maziko", threshold=0.5, phases=1, with_trace=True)
    assert len(result) == 3
    conf, fired, trace = result
    assert isinstance(trace, list)
    assert len(trace) >= 10  # ~12 frames in 1 s of 16 kHz
    # conf is the unrounded max; trace values are rounded to 4 dp -> loose tolerance.
    assert conf == pytest.approx(max(trace), abs=1e-4)
    assert all(0.0 <= v <= 1.0 for v in trace)
    assert isinstance(fired, bool)


def test_score_wake_clip_without_trace_is_two_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openwakeword(monkeypatch, _LevelModel)
    clip = np.full(16000, 0.3, dtype=np.float32)
    result = score_wake_clip(clip, 16000, "maziko", threshold=0.5, phases=1)
    assert len(result) == 2


def test_score_wake_clip_empty_with_trace_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openwakeword(monkeypatch, _LevelModel)
    conf, fired, trace = score_wake_clip(
        np.zeros(0, dtype=np.float32), 16000, "maziko", with_trace=True
    )
    assert (conf, fired, trace) == (0.0, False, [])


# --------------------------------------------------------------------------- #
# 4. wake_gain_sweep action — confidence rises with gain (the smoking gun)    #
# --------------------------------------------------------------------------- #
def test_wake_gain_sweep_confidence_rises_with_gain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_openwakeword(monkeypatch, _LevelModel)
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "my_stt_tts.wake.wake_model_for", lambda *_a, **_k: str(tmp_path / "m.onnx")
    )
    (tmp_path / "m.onnx").write_bytes(b"x")  # model "exists"
    # A SAVED quiet clip, addressed by hash.
    quiet = np.full(32000, 0.02, dtype=np.float32)
    _write_wav(tmp_path / "20260620-000000-wake-server-maziko-abcd1234.wav", quiet)

    cfg = Config(anthropic_api_key="sk-test", wake_threshold=0.5, wake_phases=1)
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_gain_sweep(cfg, "maziko", "abcd1234", [1.0, 2.0, 4.0, 8.0, 16.0])

    result = next(e for e in _drain(sub) if e["type"] == "wake_gain_sweep_result")
    assert result["word"] == "maziko"
    assert result["hash"] == "abcd1234"
    points = result["points"]
    assert [p["gain"] for p in points] == [1.0, 2.0, 4.0, 8.0, 16.0]
    confidences = [p["confidence"] for p in points]
    # The smoking gun: confidence climbs monotonically (non-decreasing) with gain.
    assert confidences == sorted(confidences)
    assert confidences[-1] > confidences[0]
    assert all(isinstance(p["fired"], bool) for p in points)


def test_wake_gain_sweep_newest_clip_when_hash_null(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """hash=None -> the newest wake recording for the word is used."""
    _install_fake_openwakeword(monkeypatch, _LevelModel)
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "my_stt_tts.wake.wake_model_for", lambda *_a, **_k: str(tmp_path / "m.onnx")
    )
    (tmp_path / "m.onnx").write_bytes(b"x")
    _write_wav(tmp_path / "20260620-000000-wake-server-maziko-aaaa1111.wav", np.full(32000, 0.3))
    cfg = Config(anthropic_api_key="sk-test", wake_threshold=0.5, wake_phases=1)
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_gain_sweep(cfg, "maziko", None, [1.0, 2.0])
    result = next(e for e in _drain(sub) if e["type"] == "wake_gain_sweep_result")
    assert len(result["points"]) == 2
    assert result["hash"] == "aaaa1111"


def test_wake_gain_sweep_no_clip_emits_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "my_stt_tts.wake.wake_model_for", lambda *_a, **_k: str(tmp_path / "m.onnx")
    )
    (tmp_path / "m.onnx").write_bytes(b"x")
    cfg = Config(anthropic_api_key="sk-test")
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_gain_sweep(cfg, "maziko", "deadbeef", [1.0, 2.0])
    result = next(e for e in _drain(sub) if e["type"] == "wake_gain_sweep_result")
    assert result["points"] == []
    assert "no saved clip" in result["message"]


def test_wake_gain_sweep_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("my_stt_tts.wake.wake_model_for", lambda *_a, **_k: "wakewords/nope.onnx")
    cfg = Config(anthropic_api_key="sk-test")
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_gain_sweep(cfg, "nope", "abcd1234", [1.0, 2.0])
    result = next(e for e in _drain(sub) if e["type"] == "wake_gain_sweep_result")
    assert result["points"] == []
    assert "unavailable" in result["message"]


# --------------------------------------------------------------------------- #
# 5. score_clip action — score the EXACT saved WAV                            #
# --------------------------------------------------------------------------- #
def test_score_clip_scores_saved_wav(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_fake_openwakeword(monkeypatch, _LevelModel)
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "my_stt_tts.wake.wake_model_for", lambda *_a, **_k: str(tmp_path / "m.onnx")
    )
    (tmp_path / "m.onnx").write_bytes(b"x")
    loud = np.full(32000, 0.6, dtype=np.float32)
    _write_wav(tmp_path / "20260620-000000-wake-server-maziko-c0ffee99.wav", loud)
    cfg = Config(anthropic_api_key="sk-test", wake_threshold=0.5, wake_phases=1)
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_score_clip(cfg, "maziko", "c0ffee99")
    result = next(e for e in _drain(sub) if e["type"] == "score_clip_result")
    assert result["word"] == "maziko"
    assert result["hash"] == "c0ffee99"
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["int16_peak"] > 0
    assert isinstance(result["fired"], bool)


def test_score_clip_no_clip_emits_zero(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "my_stt_tts.wake.wake_model_for", lambda *_a, **_k: str(tmp_path / "m.onnx")
    )
    (tmp_path / "m.onnx").write_bytes(b"x")
    cfg = Config(anthropic_api_key="sk-test")
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_score_clip(cfg, "maziko", "deadbeef")
    result = next(e for e in _drain(sub) if e["type"] == "score_clip_result")
    assert result["confidence"] == 0.0
    assert result["int16_peak"] == 0
    assert "no saved clip" in result["message"]


# --------------------------------------------------------------------------- #
# 6. event emitter shapes                                                     #
# --------------------------------------------------------------------------- #
def test_wake_gain_sweep_result_emitter_shape() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.wake_gain_sweep_result(
        word="maziko",
        hash="abcd1234",
        points=[
            {"gain": 1.0, "confidence": 0.01, "fired": False},
            {"gain": 8.0, "confidence": 0.7, "fired": True},
        ],
        message="swept 2 gains",
    )
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["type"] == "wake_gain_sweep_result"
    assert evt["word"] == "maziko"
    assert evt["hash"] == "abcd1234"
    assert evt["points"][1] == {"gain": 8.0, "confidence": 0.7, "fired": True}
    assert evt["message"] == "swept 2 gains"


def test_score_clip_result_emitter_shape() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.score_clip_result(
        word="nexus",
        hash="c0ffee99",
        confidence=0.42,
        fired=False,
        int16_peak=655,
        message="nexus: confidence 0.42",
    )
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["type"] == "score_clip_result"
    assert evt["word"] == "nexus"
    assert evt["hash"] == "c0ffee99"
    assert evt["confidence"] == 0.42
    assert evt["fired"] is False
    assert evt["int16_peak"] == 655


def test_wake_test_result_carries_score_trace_and_int16() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    stats = audio.capture_stats(np.full(16000, 0.05, dtype=np.float32), 16000)
    bus.wake_test_result(
        word="maziko",
        source="server",
        confidence=0.2,
        fired=False,
        message="maziko: confidence 0.20",
        stats=stats,
        score_trace=[0.01, 0.05, 0.2, 0.1],
    )
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["score_trace"] == [0.01, 0.05, 0.2, 0.1]
    assert evt["int16_peak"] == stats["int16_peak"]
    assert evt["expected_level"] in ("low", "ok", "high")


def test_mic_check_result_carries_int16_fields() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    stats = audio.capture_stats(np.full(16000, 0.5, dtype=np.float32), 16000)
    bus.mic_check_result(
        source="server",
        peak=0.5,
        level=50,
        rms=0.5,
        duration_s=1.0,
        sample_rate=16000,
        levels=[0.5],
        processing={"agc": False, "ns": False, "ec": False, "gain": 2.0},
        hash="abcd1234",
        wav_url="/recordings/x.wav",
        message="ok",
        stats=stats,
    )
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["int16_peak"] == stats["int16_peak"]
    assert evt["int16_rms"] == stats["int16_rms"]
    assert evt["expected_level"] == "ok"  # 0.5 float -> ~16383 int16, in band


def test_result_emitters_have_metric_defaults_without_stats() -> None:
    """A result emitted WITHOUT a stats dict still carries the full metric field set."""
    bus = EventBus()
    sub = bus.subscribe()
    bus.score_clip_result(word="x", hash="", confidence=0.0, fired=False, int16_peak=0, message="")
    # mic_check_result with stats=None -> defaults present.
    bus.mic_check_result(
        source="server",
        peak=0.0,
        level=0,
        rms=0.0,
        duration_s=0.0,
        sample_rate=16000,
        levels=[],
        processing={},
        hash="",
        wav_url="",
        message="",
    )
    events = _drain(sub)
    mc = next(e for e in events if e["type"] == "mic_check_result")
    assert mc["int16_peak"] == 0
    assert mc["expected_level"] == "low"
    assert mc["snr_db"] is None
    assert mc["lufs"] is None


# --------------------------------------------------------------------------- #
# 7. capture_stats: int16, expected_level, crest, dc, true_peak, snr, lufs    #
# --------------------------------------------------------------------------- #
def test_capture_stats_int16_magnitudes() -> None:
    stats = audio.capture_stats(np.full(16000, 0.5, dtype=np.float32), 16000)
    # 0.5 float * 32767 -> ~16383 on the int16 scale.
    assert stats["int16_peak"] == pytest.approx(16383, abs=2)
    assert stats["int16_rms"] == pytest.approx(16383, abs=2)


@pytest.mark.parametrize(
    ("float_peak", "expected"),
    [
        (0.02, "low"),  # ~655 int16 < 2000 -> low (the dead-wake symptom)
        (0.3, "ok"),  # ~9830 int16 in band
        (0.99, "high"),  # ~32440 int16 > 30000 -> clipping
    ],
)
def test_capture_stats_expected_level_classification(float_peak: float, expected: str) -> None:
    # Use a tone so peak != rms and the level is driven by the peak as intended.
    t = np.linspace(0, 50, 16000)
    clip = (np.sin(t) * float_peak).astype(np.float32)
    assert audio.capture_stats(clip, 16000)["expected_level"] == expected


def test_capture_stats_crest_db_math() -> None:
    # A full-scale square wave has peak == rms -> crest 0 dB.
    square = np.concatenate([np.full(8000, 0.5), np.full(8000, -0.5)]).astype(np.float32)
    assert audio.capture_stats(square, 16000)["crest_db"] == pytest.approx(0.0, abs=0.01)
    # A sine has crest ~3.01 dB (peak/rms = sqrt(2)).
    sine = (np.sin(np.linspace(0, 100 * np.pi, 16000)) * 0.5).astype(np.float32)
    assert audio.capture_stats(sine, 16000)["crest_db"] == pytest.approx(3.01, abs=0.1)


def test_capture_stats_dc_offset() -> None:
    biased = np.full(16000, 0.25, dtype=np.float32)
    assert audio.capture_stats(biased, 16000)["dc_offset"] == pytest.approx(0.25, abs=1e-4)
    centred = (np.sin(np.linspace(0, 100 * np.pi, 16000)) * 0.5).astype(np.float32)
    assert audio.capture_stats(centred, 16000)["dc_offset"] == pytest.approx(0.0, abs=1e-3)


def test_capture_stats_true_peak_db() -> None:
    # A full-scale (1.0) signal is ~0 dBFS true peak; a -6 dB (0.5) signal ~ -6 dBFS.
    full = (np.sin(np.linspace(0, 50 * np.pi, 16000))).astype(np.float32)
    half = full * 0.5
    tp_full = audio.capture_stats(full, 16000)["true_peak_db"]
    tp_half = audio.capture_stats(half, 16000)["true_peak_db"]
    assert tp_full == pytest.approx(0.0, abs=0.5)
    assert tp_half == pytest.approx(-6.0, abs=1.0)
    assert tp_full > tp_half


def test_capture_stats_snr_db_speech_vs_noise() -> None:
    """A clip of loud 'speech' windows over a quiet floor has a positive, finite SNR."""
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal(16000) * 0.005).astype(np.float32)  # quiet floor
    speech = noise.copy()
    speech[4000:12000] += (np.sin(np.linspace(0, 200, 8000)) * 0.4).astype(np.float32)
    snr = audio.capture_stats(speech, 16000)["snr_db"]
    assert snr is not None
    assert snr > 10.0  # loud speech well above the noise floor


def test_capture_stats_snr_db_none_on_pure_silence() -> None:
    assert audio.capture_stats(np.zeros(16000, dtype=np.float32), 16000)["snr_db"] is None


def test_capture_stats_snr_db_none_when_too_short() -> None:
    # Fewer than 2 windows of 20 ms -> indeterminable.
    assert audio.capture_stats(np.full(100, 0.3, dtype=np.float32), 16000)["snr_db"] is None


_HAVE_PYLOUDNORM = importlib.util.find_spec("pyloudnorm") is not None


@pytest.mark.skipif(_HAVE_PYLOUDNORM, reason="pyloudnorm IS installed; tested in the gated case")
def test_capture_stats_lufs_none_without_pyloudnorm() -> None:
    """Core-clean: without the `debug` extra, lufs degrades to None (no crash)."""
    clip = (np.sin(np.linspace(0, 400, 32000)) * 0.5).astype(np.float32)
    assert audio.capture_stats(clip, 16000)["lufs"] is None


@pytest.mark.skipif(not _HAVE_PYLOUDNORM, reason="needs the `debug` extra (pyloudnorm)")
def test_capture_stats_lufs_when_pyloudnorm_present() -> None:
    """With the `debug` extra, lufs is a finite negative dB loudness for real audio."""
    clip = (np.sin(np.linspace(0, 400 * np.pi, 32000)) * 0.5).astype(np.float32)
    lufs = audio.capture_stats(clip, 16000)["lufs"]
    assert lufs is not None
    assert -60.0 < lufs < 0.0


def test_capture_stats_keeps_base_fields() -> None:
    """The original capture summary fields are preserved (back-compat)."""
    stats = audio.capture_stats(np.full(16000, 0.3, dtype=np.float32), 16000)
    assert stats["sample_rate"] == 16000
    assert stats["samples"] == 16000
    assert stats["duration_s"] == pytest.approx(1.0)
    assert "rms" in stats and "peak" in stats


# --------------------------------------------------------------------------- #
# 8. action labels                                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("wake_gain_sweep", "clicked WAKE GAIN SWEEP"),
        ("score_clip", "clicked SCORE CLIP"),
    ],
)
def test_new_action_labels(name: str, label: str) -> None:
    assert main_mod._AudioDebug.action_label(name) == label


# --------------------------------------------------------------------------- #
# 9. WakeWord.detect smoke (gain reaches the model via to_int16_pcm)          #
# --------------------------------------------------------------------------- #
def test_wakeword_detect_scales_to_int16(monkeypatch: pytest.MonkeyPatch) -> None:
    """A gained float frame reaches the model as a non-zero int16 PCM (sanity)."""
    _install_fake_openwakeword(monkeypatch, _LevelModel)
    detector = WakeWord("wakewords/maziko.onnx", 0.5, phases=1)
    # Drive a full 1280-sample frame of moderate amplitude through detect.
    frame = audio.apply_gain(np.full(1280, 0.05, dtype=np.float32), 8.0)
    detector.detect(frame)
    assert detector.last_score > 0.0  # the model saw real energy (not truncated zeros)
