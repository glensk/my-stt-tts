"""Wake-word TEST diagnostic: ``score_wake_clip`` + the GUI ``wake_test`` action.

The user asked for a way to confirm a wake word would actually fire. This covers:

* :func:`wake.score_wake_clip` — load the model for an ARBITRARY word (not the
  configured one), resample the clip to 16 kHz, feed it through the REAL phase-diverse
  :meth:`WakeWord.detect` path frame-by-frame, and return ``(max score, fired)``.
  Verified with a fake openWakeWord model (deterministic, no wheel) for the plumbing,
  and — when openWakeWord + ``say`` are present — against a REAL synthesized clip so a
  spoken word scores high and white noise scores ~0 (mirrors ``detect``).
* The server + browser ``wake_test`` actions: record / build PCM, score, save the
  16 kHz clip as a WAV, and emit the ``wake_test_result`` event (mic + model mocked).
* The missing-model path (clear message, no crash).
* The friendly EVENT-LOG action labels.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring,import-outside-toplevel

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import types
import wave
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from my_stt_tts import __main__ as main_mod
from my_stt_tts import audio
from my_stt_tts.config import Config
from my_stt_tts.events import EventBus
from my_stt_tts.wake import score_wake_clip


# --------------------------------------------------------------------------- #
# Fake openWakeWord plumbing (shared with the other wake tests' pattern)      #
# --------------------------------------------------------------------------- #
def _install_fake_openwakeword(monkeypatch: pytest.MonkeyPatch, model_cls: type) -> None:
    pkg = types.ModuleType("openwakeword")
    mod = types.ModuleType("openwakeword.model")
    mod.Model = model_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.model", mod)


class _EnergyModel:
    """Scores high on loud int16 PCM, ~0 on silence — a deterministic stand-in."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        score = 0.9 if int(np.max(np.abs(frame))) > 1000 else 0.001
        return {"maziko": score}


# --------------------------------------------------------------------------- #
# score_wake_clip — plumbing (resample, frame-by-frame detect, max + threshold)#
# --------------------------------------------------------------------------- #
def test_score_wake_clip_high_on_loud_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openwakeword(monkeypatch, _EnergyModel)
    # ~2 s of a loud tone at 16 kHz -> the fake model scores 0.9 per frame.
    clip = (np.sin(np.linspace(0, 400, 32000)) * 0.5).astype(np.float32)
    confidence, fired = score_wake_clip(clip, 16000, "maziko", threshold=0.4, phases=4)
    assert confidence == pytest.approx(0.9)
    assert fired is True


def test_score_wake_clip_zero_on_silence(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openwakeword(monkeypatch, _EnergyModel)
    clip = np.zeros(32000, dtype=np.float32)
    confidence, fired = score_wake_clip(clip, 16000, "maziko", threshold=0.4)
    assert confidence == pytest.approx(0.001)
    assert fired is False


def test_score_wake_clip_resamples_non_16k(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 48 kHz clip is resampled to 16 kHz before scoring (the model needs 16 kHz)."""
    seen_lengths: list[int] = []

    class FrameLenModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, frame: np.ndarray) -> dict[str, float]:
            seen_lengths.append(int(frame.size))
            return {"maziko": 0.2}

    _install_fake_openwakeword(monkeypatch, FrameLenModel)
    clip48 = (np.sin(np.linspace(0, 100, 96000)) * 0.4).astype(np.float32)  # 2 s @ 48 kHz
    confidence, _ = score_wake_clip(clip48, 48000, "maziko", threshold=0.4, phases=1)
    assert confidence == pytest.approx(0.2)
    # Every scored frame is exactly 1280 samples (16 kHz / 80 ms) -> resample worked.
    assert seen_lengths and all(n == 1280 for n in seen_lengths)
    # 2 s of 16 kHz audio is ~25 frames of 1280 samples.
    assert len(seen_lengths) >= 20


def test_score_wake_clip_tracks_max_over_clip(monkeypatch: pytest.MonkeyPatch) -> None:
    """confidence is the MAX last_score over the clip, not the last frame's."""
    scores = iter([0.1, 0.8, 0.2, 0.05])

    class RampModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, _frame: np.ndarray) -> dict[str, float]:
            return {"maziko": next(scores, 0.0)}

    _install_fake_openwakeword(monkeypatch, RampModel)
    clip = (np.sin(np.linspace(0, 50, 16000)) * 0.4).astype(np.float32)
    confidence, fired = score_wake_clip(clip, 16000, "maziko", threshold=0.5, phases=1)
    assert confidence == pytest.approx(0.8)
    assert fired is True


def test_score_wake_clip_empty_returns_zero() -> None:
    confidence, fired = score_wake_clip(np.zeros(0, dtype=np.float32), 16000, "maziko")
    assert (confidence, fired) == (0.0, False)


def test_score_wake_clip_missing_model_returns_zero(tmp_path: Path) -> None:
    confidence, fired = score_wake_clip(
        np.ones(16000, dtype=np.float32),
        16000,
        "doesnotexist",
        wakewords_dir=str(tmp_path),
    )
    assert (confidence, fired) == (0.0, False)


# --------------------------------------------------------------------------- #
# REAL model + `say` integration (skipped in core: openWakeWord is the `wake`  #
# extra and `say` is macOS-only). Runs score_wake_clip end-to-end through the   #
# ACTUAL maziko/nexus ONNX models and asserts white noise scores ~0 and does    #
# NOT fire — i.e. the scoring path mirrors the live `detect` loop and the model  #
# isn't trigger-happy. (Note: macOS `say` is a synthetic TTS voice the models    #
# were NOT trained on, so it does not reliably FIRE them — only a real human     #
# "maziko"/"nexus" does, which a unit test can't synthesize. The fake-model       #
# tests above cover the high-score / max-tracking / threshold logic.)            #
# --------------------------------------------------------------------------- #
def _have_openwakeword() -> bool:
    import importlib.util

    return importlib.util.find_spec("openwakeword") is not None


def _say_to_clip(word: str) -> np.ndarray:
    """Synthesize a spoken ``word`` to a 16 kHz mono float32 clip via macOS ``say``."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "say.wav"
        # AIFF -> 16 kHz WAV via afconvert (both macOS built-ins).
        aiff = Path(td) / "say.aiff"
        subprocess.run(["say", "-o", str(aiff), word], check=True)
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(aiff), str(wav)],
            check=True,
        )
        with wave.open(str(wav), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return data


@pytest.mark.skipif(
    not (_have_openwakeword() and shutil.which("say") and shutil.which("afconvert")),
    reason="needs the openWakeWord `wake` extra and macOS say/afconvert",
)
@pytest.mark.parametrize("word", ["maziko", "nexus"])
def test_score_wake_clip_real_model_noise_does_not_fire(word: str) -> None:
    if not Path(f"wakewords/{word}.onnx").is_file():
        pytest.skip(f"no trained {word}.onnx model")
    # End-to-end through the REAL ONNX model: a spoken clip scores a finite,
    # in-range confidence (the path runs, mirrors detect)…
    speech = _say_to_clip(word)
    spoken_conf, _ = score_wake_clip(speech, 16000, word, threshold=0.4, phases=8)
    assert 0.0 <= spoken_conf <= 1.0
    # …and white noise is ~silent to the model and must NOT fire.
    noise = (np.random.default_rng(0).standard_normal(32000) * 0.05).astype(np.float32)
    noise_conf, noise_fired = score_wake_clip(noise, 16000, word, threshold=0.4, phases=8)
    assert noise_conf < 0.3
    assert noise_fired is False


# --------------------------------------------------------------------------- #
# Server wake_test action: record -> score -> save WAV -> emit event           #
# --------------------------------------------------------------------------- #
def _drain(sub: Any) -> list[dict[str, Any]]:
    import queue

    out: list[dict[str, Any]] = []
    while True:
        try:
            out.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    return out


def test_run_wake_test_server_records_scores_saves_emits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    clip = (np.sin(np.linspace(0, 100, 32000)) * 0.4).astype(np.float32)
    monkeypatch.setattr(audio, "record_fixed", lambda *_a, **_k: (clip, 16000))
    monkeypatch.setattr("my_stt_tts.wake.score_wake_clip", lambda *_a, **_k: (0.73, True))

    wav = tmp_path / "wake-test-maziko-server.wav"
    monkeypatch.setattr(main_mod, "_wake_test_wav_path", lambda *_a: str(wav))

    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_test_server(cfg, "maziko")

    events = _drain(sub)
    result = next(e for e in events if e["type"] == "wake_test_result")
    assert result["word"] == "maziko"
    assert result["source"] == "server"
    assert result["confidence"] == pytest.approx(0.73)
    assert result["fired"] is True
    assert "confidence 0.73" in result["message"]
    assert result["wav_path"] == str(wav)
    # The 16 kHz clip was actually written and is a readable mono WAV.
    assert wav.is_file()
    with wave.open(str(wav), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getnframes() > 0


def test_run_wake_test_server_capture_error_emits_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(anthropic_api_key="sk-test")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no mic")

    monkeypatch.setattr(audio, "record_fixed", _boom)
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_test_server(cfg, "maziko")
    result = next(e for e in _drain(sub) if e["type"] == "wake_test_result")
    assert result["confidence"] == 0.0
    assert result["fired"] is False
    assert "microphone error" in result["message"]


def test_run_wake_test_server_missing_model_emits_clear_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    # No model file for this word -> a clear "unavailable" message, no crash, no record.
    monkeypatch.setattr("my_stt_tts.wake.wake_model_for", lambda *_a, **_k: "wakewords/nope.onnx")
    called = {"recorded": False}

    def _should_not_run(*_a: Any, **_k: Any) -> Any:
        called["recorded"] = True
        raise AssertionError("must not record when the model is missing")

    monkeypatch.setattr(audio, "record_fixed", _should_not_run)
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_test_server(cfg, "nope")
    result = next(e for e in _drain(sub) if e["type"] == "wake_test_result")
    assert result["confidence"] == 0.0
    assert result["fired"] is False
    assert "unavailable" in result["message"]
    assert called["recorded"] is False


# --------------------------------------------------------------------------- #
# Browser wake_test action: build PCM -> score -> save WAV -> emit             #
# --------------------------------------------------------------------------- #
def test_run_wake_test_browser_scores_saves_emits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    monkeypatch.setattr("my_stt_tts.wake.score_wake_clip", lambda *_a, **_k: (0.31, False))
    wav = tmp_path / "wake-test-nexus-browser.wav"
    monkeypatch.setattr(main_mod, "_wake_test_wav_path", lambda *_a: str(wav))

    pcm = list((np.sin(np.linspace(0, 100, 32000)) * 0.4).astype(np.float32))
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_test_browser(cfg, "nexus", pcm, 16000)

    result = next(e for e in _drain(sub) if e["type"] == "wake_test_result")
    assert result["word"] == "nexus"
    assert result["source"] == "browser"
    assert result["confidence"] == pytest.approx(0.31)
    assert result["fired"] is False
    assert "not detected" in result["message"]
    assert wav.is_file()


def test_run_wake_test_browser_missing_model(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    monkeypatch.setattr("my_stt_tts.wake.wake_model_for", lambda *_a, **_k: "wakewords/nope.onnx")
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_test_browser(cfg, "nope", [0.0, 0.1, 0.2], 16000)
    result = next(e for e in _drain(sub) if e["type"] == "wake_test_result")
    assert result["confidence"] == 0.0
    assert "unavailable" in result["message"]


# --------------------------------------------------------------------------- #
# wake_test_result event emitter shape                                         #
# --------------------------------------------------------------------------- #
def test_wake_test_result_emitter_shape() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.wake_test_result(
        word="maziko",
        source="server",
        confidence=0.37,
        fired=False,
        message="maziko: confidence 0.37 — not detected",
        wav_path="/tmp/x.wav",
    )
    evt = json.loads(sub.get(timeout=1.0))
    assert evt == {
        "type": "wake_test_result",
        "word": "maziko",
        "source": "server",
        "confidence": 0.37,
        "fired": False,
        "message": "maziko: confidence 0.37 — not detected",
        "wav_path": "/tmp/x.wav",
    }


# --------------------------------------------------------------------------- #
# Friendly EVENT-LOG action labels                                            #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("ptt", "clicked PUSH-TO-TALK"),
        ("wake_start", "clicked START WAKE"),
        ("wake_stop", "clicked STOP WAKE"),
        ("mic_test", "clicked TEST SERVER MIC"),
        ("mic_record_replay", "clicked RECORD & PLAY · SERVER"),
        ("live_audio", "clicked LIVE AUDIO"),
        ("reset", "clicked RESET"),
        ("wake_test", "clicked WAKE TEST"),
        ("turn", "submitted a turn"),
        ("frobnicate", "clicked FROBNICATE"),  # unknown -> "clicked <NAME>"
    ],
)
def test_action_label_mapping(name: str, label: str) -> None:
    assert main_mod._AudioDebug.action_label(name) == label


def test_action_emits_friendly_message_keeps_stage(capsys: pytest.CaptureFixture[str]) -> None:
    dbg = main_mod._AudioDebug(True)
    with patch.object(main_mod.bus, "debug") as busdebug:
        dbg.action("ptt")
    # The GUI EVENT LOG message is the friendly label…
    assert busdebug.call_args.args[0] == "clicked PUSH-TO-TALK"
    # …while the structured machine field is still stage="action:ptt".
    assert busdebug.call_args.kwargs["stage"] == "action:ptt"
    # stderr keeps the [audio:action:ptt] prefix for grep + shows the friendly text.
    err = capsys.readouterr().err
    assert "[audio:action:ptt]" in err
    assert "clicked PUSH-TO-TALK" in err


def test_action_drops_huge_pcm_field() -> None:
    dbg = main_mod._AudioDebug(True)
    with patch.object(main_mod.bus, "debug") as busdebug:
        dbg.action("wake_test", word="maziko", source="browser", pcm=[0.1] * 32000)
    # The thousands-of-floats pcm payload is NOT logged; the small fields survive.
    assert "pcm" not in busdebug.call_args.kwargs
    assert busdebug.call_args.kwargs["word"] == "maziko"
    assert busdebug.call_args.kwargs["source"] == "browser"
    assert busdebug.call_args.args[0].startswith("clicked WAKE TEST")
