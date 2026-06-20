"""Mic/wake DEBUG backend overhaul: recordings infra, the ``/recordings/`` route,
software ``mic_gain``, the unified ``mic_check`` action, the extended ``wake_test``
result shape, ``play_recording``, and the LLM media-honesty system prompt.

Everything that would touch a real microphone / model / speaker / the network is
mocked — only pure logic, the on-disk WAV archive (under a tmp recordings dir), and
the stdlib HTTP route are exercised.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring,import-outside-toplevel,redefined-outer-name

from __future__ import annotations

import http.client
import json
import queue
import threading
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from my_stt_tts import __main__ as main_mod
from my_stt_tts import audio
from my_stt_tts.config import Config, ConfigError
from my_stt_tts.events import EventBus
from my_stt_tts.webui import WebUI


def _drain(sub: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while True:
        try:
            out.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    return out


# --------------------------------------------------------------------------- #
# Recordings infra: save_recording -> path + hash8 + wav_url, real WAV on disk #
# --------------------------------------------------------------------------- #
def test_save_recording_writes_16k_wav_with_hash_and_url(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    clip = (np.sin(np.linspace(0, 100, 32000)) * 0.4).astype(np.float32)
    path, hash8, wav_url = audio.save_recording(clip, 16000, kind="mic", source="server")
    assert len(hash8) == 8 and all(c in "0123456789abcdef" for c in hash8)
    # Mic-check clips stay flat: <ts>-<source>-<hash>.wav directly under recordings/.
    assert wav_url == f"/recordings/{Path(path).name}"
    assert Path(path).name.endswith(f"-server-{hash8}.wav")
    assert Path(path).parent == tmp_path
    # A readable mono 16 kHz WAV was written.
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnchannels() == 1
        assert wf.getnframes() > 0


def test_save_recording_resamples_to_16k_and_names_word(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    clip48 = (np.sin(np.linspace(0, 100, 96000)) * 0.4).astype(np.float32)  # 2 s @ 48 kHz
    path, hash8, wav_url = audio.save_recording(
        clip48, 48000, kind="wake", source="browser", word="maziko"
    )
    # Wake clips are kept as training data in a PER-WORD subfolder, named
    # <ts>-<source>-<hash>.wav (the word is the folder, not in the filename).
    assert Path(path).parent == tmp_path / "wake" / "maziko"
    assert Path(path).name.endswith(f"-browser-{hash8}.wav")
    assert wav_url == f"/recordings/wake/maziko/{Path(path).name}"
    with wave.open(path, "rb") as wf:
        assert wf.getframerate() == 16000
        # ~2 s of 16 kHz audio (resampled from 48 kHz).
        assert 30000 < wf.getnframes() < 34000


def test_save_recording_hash_is_content_addressed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    clip = (np.sin(np.linspace(0, 100, 16000)) * 0.4).astype(np.float32)
    _p1, h1, _u1 = audio.save_recording(clip, 16000, kind="mic", source="server")
    _p2, h2, _u2 = audio.save_recording(clip, 16000, kind="mic", source="server")
    assert h1 == h2  # same PCM -> same hash
    other = (np.sin(np.linspace(0, 50, 16000)) * 0.4).astype(np.float32)
    _p3, h3, _u3 = audio.save_recording(other, 16000, kind="mic", source="server")
    assert h3 != h1


def test_save_recording_disk_error_returns_empty_path_but_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: "/no/such/dir/at/all")

    def _boom(*_a: Any, **_k: Any) -> None:
        raise OSError("read-only fs")

    monkeypatch.setattr("os.makedirs", _boom)
    clip = (np.sin(np.linspace(0, 100, 16000)) * 0.4).astype(np.float32)
    path, hash8, wav_url = audio.save_recording(clip, 16000, kind="mic", source="server")
    assert path == ""  # write failed -> empty path
    assert len(hash8) == 8  # hash + url still computed for the GUI
    assert wav_url.startswith("/recordings/")


# --------------------------------------------------------------------------- #
# compute_levels + apply_gain (the level graph + clip-protected software gain)  #
# --------------------------------------------------------------------------- #
def test_compute_levels_returns_fixed_window_count() -> None:
    clip = (np.sin(np.linspace(0, 500, 32000)) * 0.5).astype(np.float32)
    levels = audio.compute_levels(clip)
    assert len(levels) == 48
    assert all(0.0 <= v <= 1.0 for v in levels)
    assert max(levels) > 0.0


def test_compute_levels_empty_is_all_zero() -> None:
    levels = audio.compute_levels(np.zeros(0, dtype=np.float32))
    assert levels == [0.0] * 48


def test_apply_gain_lifts_quiet_and_clips_hot() -> None:
    quiet = np.full(100, 0.1, dtype=np.float32)
    gained = audio.apply_gain(quiet, 2.0)
    assert np.allclose(gained, 0.2)
    # A hot clip is hard-clipped to ±1.0, never wrapped.
    hot = np.array([0.8, -0.9, 0.6], dtype=np.float32)
    clipped = audio.apply_gain(hot, 4.0)
    assert clipped.max() <= 1.0 and clipped.min() >= -1.0
    assert clipped[0] == pytest.approx(1.0)
    assert clipped[1] == pytest.approx(-1.0)


def test_apply_gain_identity_at_one() -> None:
    clip = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    assert np.array_equal(audio.apply_gain(clip, 1.0), clip)


# --------------------------------------------------------------------------- #
# mic_gain config: default, env, validate bounds                              #
# --------------------------------------------------------------------------- #
def test_mic_gain_default_is_two() -> None:
    assert Config().mic_gain == 2.0


def test_mic_gain_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIC_GAIN", "3.5")
    assert Config.from_env().mic_gain == 3.5


@pytest.mark.parametrize("bad", [0.0, -1.0, 10.5, 100.0])
def test_mic_gain_validate_rejects_out_of_range(bad: float) -> None:
    cfg = Config(anthropic_api_key="sk-test", mic_gain=bad)
    with pytest.raises(ConfigError, match="mic_gain"):
        cfg.validate()


def test_mic_gain_validate_accepts_in_range() -> None:
    Config(anthropic_api_key="sk-test", mic_gain=2.0).validate()
    Config(anthropic_api_key="sk-test", mic_gain=10.0).validate()


# --------------------------------------------------------------------------- #
# mic_check action — SERVER: record -> gain -> save -> mic_check_result        #
# --------------------------------------------------------------------------- #
def test_run_mic_check_server_full_result_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test", mic_gain=2.0)
    # A quiet ~0.1-peak clip at the device rate; the 2.0× gain lifts it to ~0.2.
    clip = np.full(32000, 0.1, dtype=np.float32)
    monkeypatch.setattr(audio, "record_fixed", lambda *_a, **_k: (clip, 16000))
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))

    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_mic_check_server(cfg)
    result = next(e for e in _drain(sub) if e["type"] == "mic_check_result")

    assert result["source"] == "server"
    assert result["peak"] == pytest.approx(0.2, abs=0.01)  # 0.1 * 2.0 gain
    assert result["level"] == round(result["peak"] * 100)
    assert result["sample_rate"] == 16000
    assert result["duration_s"] == pytest.approx(2.0, abs=0.1)
    assert isinstance(result["levels"], list) and len(result["levels"]) == 48
    assert result["processing"] == {"agc": False, "ns": False, "ec": False, "gain": 2.0}
    assert len(result["hash"]) == 8
    assert result["wav_url"].startswith("/recordings/")
    # The gained clip was actually archived.
    assert any(tmp_path.iterdir())


def test_run_mic_check_server_capture_error_emits_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(anthropic_api_key="sk-test")

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("no mic")

    monkeypatch.setattr(audio, "record_fixed", _boom)
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_mic_check_server(cfg)
    result = next(e for e in _drain(sub) if e["type"] == "mic_check_result")
    assert result["peak"] == 0.0
    assert result["level"] == 0
    assert result["hash"] == ""
    assert "microphone error" in result["message"]
    assert result["processing"]["gain"] == cfg.mic_gain


# --------------------------------------------------------------------------- #
# mic_check action — BROWSER: save + analyse the posted clip, no server gain    #
# --------------------------------------------------------------------------- #
def test_run_mic_check_browser_result_shape_and_processing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    pcm = list((np.sin(np.linspace(0, 100, 32000)) * 0.4).astype(np.float32))
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_mic_check_browser(cfg, pcm, 16000, {"agc": True, "ns": False, "ec": True})
    result = next(e for e in _drain(sub) if e["type"] == "mic_check_result")
    assert result["source"] == "browser"
    assert result["peak"] > 0.0
    assert len(result["levels"]) == 48
    # Browser flags pass through; no server gain (gain == 1.0).
    assert result["processing"] == {"agc": True, "ns": False, "ec": True, "gain": 1.0}
    assert result["wav_url"].startswith("/recordings/")


def test_run_mic_check_browser_missing_flags_are_null(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    pcm = list((np.sin(np.linspace(0, 100, 16000)) * 0.4).astype(np.float32))
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_mic_check_browser(cfg, pcm, 16000, None)
    result = next(e for e in _drain(sub) if e["type"] == "mic_check_result")
    assert result["processing"] == {"agc": None, "ns": None, "ec": None, "gain": 1.0}


# --------------------------------------------------------------------------- #
# wake_test extended shape: peak/level/levels/processing/hash/wav_url present   #
# --------------------------------------------------------------------------- #
def test_run_wake_test_server_emits_extended_level_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test", mic_gain=2.0)
    clip = np.full(32000, 0.1, dtype=np.float32)
    monkeypatch.setattr(audio, "record_fixed", lambda *_a, **_k: (clip, 16000))
    monkeypatch.setattr(
        "my_stt_tts.wake.score_wake_clip",
        lambda *_a, **k: (0.73, True, [0.2, 0.73]) if k.get("with_trace") else (0.73, True),
    )
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    monkeypatch.setattr(main_mod, "_wake_test_wav_path", lambda *_a: str(tmp_path / "legacy.wav"))

    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_test_server(cfg, "maziko")
    result = next(e for e in _drain(sub) if e["type"] == "wake_test_result")

    # Legacy fields preserved…
    assert result["word"] == "maziko"
    assert result["confidence"] == pytest.approx(0.73)
    assert result["fired"] is True
    # …plus the new level-meter fields.
    assert result["peak"] == pytest.approx(0.2, abs=0.01)  # gain-lifted
    assert result["level"] == round(result["peak"] * 100)
    assert len(result["levels"]) == 48
    assert result["processing"] == {"agc": False, "ns": False, "ec": False, "gain": 2.0}
    assert len(result["hash"]) == 8
    assert result["wav_url"].startswith("/recordings/")


def test_run_wake_test_browser_emits_processing_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    monkeypatch.setattr(
        "my_stt_tts.wake.score_wake_clip",
        lambda *_a, **k: (0.31, False, [0.05, 0.31]) if k.get("with_trace") else (0.31, False),
    )
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    monkeypatch.setattr(main_mod, "_wake_test_wav_path", lambda *_a: str(tmp_path / "legacy.wav"))
    pcm = list((np.sin(np.linspace(0, 100, 32000)) * 0.4).astype(np.float32))
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._run_wake_test_browser(cfg, "nexus", pcm, 16000, {"agc": False, "ns": True})
    result = next(e for e in _drain(sub) if e["type"] == "wake_test_result")
    assert result["confidence"] == pytest.approx(0.31)
    assert result["processing"]["ns"] is True
    assert result["processing"]["gain"] == 1.0  # no server gain on a browser clip
    assert len(result["levels"]) == 48
    assert result["wav_url"].startswith("/recordings/")


# --------------------------------------------------------------------------- #
# play_recording: locate a saved WAV by hash and play it (audio.play mocked)    #
# --------------------------------------------------------------------------- #
def test_play_recording_plays_saved_clip(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    clip = (np.sin(np.linspace(0, 100, 16000)) * 0.4).astype(np.float32)
    _path, hash8, _url = audio.save_recording(clip, 16000, kind="mic", source="server")

    played: dict[str, Any] = {}
    monkeypatch.setattr(audio, "play", lambda c, r, *a, **k: played.update(rate=r, n=len(c)))
    bus = EventBus()
    with patch.object(main_mod, "bus", bus):
        main_mod._play_recording(hash8)
    assert played["rate"] == 16000
    assert played["n"] > 0


def test_play_recording_unknown_hash_logs_and_does_not_play(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    called = {"played": False}
    monkeypatch.setattr(audio, "play", lambda *a, **k: called.update(played=True))
    bus = EventBus()
    sub = bus.subscribe()
    with patch.object(main_mod, "bus", bus):
        main_mod._play_recording("deadbeef")
    assert called["played"] is False
    assert any("no saved recording" in e.get("message", "") for e in _drain(sub))


# --------------------------------------------------------------------------- #
# mic_check_result emitter shape                                              #
# --------------------------------------------------------------------------- #
def test_mic_check_result_emitter_shape() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.mic_check_result(
        source="server",
        peak=0.42,
        level=42,
        rms=0.2,
        duration_s=2.0,
        sample_rate=16000,
        levels=[0.1, 0.42, 0.3],
        processing={"agc": False, "ns": False, "ec": False, "gain": 2.0},
        hash="abcd1234",
        wav_url="/recordings/x.wav",
        message="Microphone OK — level 42%",
    )
    evt = json.loads(sub.get(timeout=1.0))
    assert evt["type"] == "mic_check_result"
    assert evt["source"] == "server"
    assert evt["peak"] == 0.42
    assert evt["level"] == 42
    assert evt["levels"] == [0.1, 0.42, 0.3]
    assert evt["processing"] == {"agc": False, "ns": False, "ec": False, "gain": 2.0}
    assert evt["hash"] == "abcd1234"
    assert evt["wav_url"] == "/recordings/x.wav"


# --------------------------------------------------------------------------- #
# GET /recordings/<file>.wav — serves the WAV, path-traversal-safe            #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def served_ui(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[int, Path]]:
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    cfg = Config(sample_rate=16000)
    ui = WebUI(cfg, on_turn=lambda _t: None, on_action=lambda _n, _d: None, port=0)
    port = ui._server.server_address[1]  # type: ignore[attr-defined]
    thread = threading.Thread(target=ui.serve_forever, daemon=True)
    thread.start()
    try:
        yield port, tmp_path
    finally:
        ui._server.shutdown()  # type: ignore[attr-defined]
        ui._server.server_close()  # type: ignore[attr-defined]


def test_recordings_route_serves_wav(served_ui: tuple[int, Path]) -> None:
    port, recdir = served_ui
    clip = (np.sin(np.linspace(0, 100, 16000)) * 0.4).astype(np.float32)
    path, _hash8, wav_url = audio.save_recording(clip, 16000, kind="mic", source="server")
    assert Path(path).parent == recdir
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", wav_url)
        resp = conn.getresponse()
        body = resp.read()
        assert resp.status == 200
        assert "audio/wav" in (resp.getheader("Content-Type") or "")
        assert body[:4] == b"RIFF"  # a real WAV container
    finally:
        conn.close()


@pytest.mark.parametrize(
    "evil",
    [
        "/recordings/../../etc/passwd",
        "/recordings/..%2f..%2fetc%2fpasswd",
        "/recordings/subdir/x.wav",
        "/recordings/notawav.txt",
    ],
)
def test_recordings_route_rejects_traversal(served_ui: tuple[int, Path], evil: str) -> None:
    port, _recdir = served_ui
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", evil)
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 404  # never escapes the recordings dir
    finally:
        conn.close()


def test_recordings_route_404_for_missing_file(served_ui: tuple[int, Path]) -> None:
    port, _recdir = served_ui
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        conn.request("GET", "/recordings/20990101-000000-mic-server-00000000.wav")
        resp = conn.getresponse()
        resp.read()
        assert resp.status == 404
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Settings carry mic_gain; apply_settings clamps it                           #
# --------------------------------------------------------------------------- #
def test_settings_dict_exposes_mic_gain() -> None:
    from my_stt_tts.webui import settings_dict

    payload = settings_dict(Config(mic_gain=3.0))
    assert payload["mic_gain"] == 3.0


def test_apply_settings_clamps_mic_gain() -> None:
    from my_stt_tts.webui import apply_settings

    cfg = Config()
    apply_settings(cfg, {"mic_gain": 99.0})
    assert cfg.mic_gain == 10.0
    apply_settings(cfg, {"mic_gain": -5.0})
    assert cfg.mic_gain == 0.01


# --------------------------------------------------------------------------- #
# LLM honesty: the system prompt forbids claiming "I'll play …"               #
# --------------------------------------------------------------------------- #
def test_system_prompt_file_states_cannot_play_media() -> None:
    text = Path("prompts/system_prompt.md").read_text(encoding="utf-8").lower()
    assert "cannot play" in text
    assert "i'll play" in text  # the forbidden phrase is explicitly named
    assert "only by the system" in text


def test_config_fallback_prompt_states_cannot_play_media() -> None:
    from my_stt_tts.config import _DEFAULT_SYSTEM_PROMPT

    low = _DEFAULT_SYSTEM_PROMPT.lower()
    assert "cannot" in low and "play" in low
    assert "i'll play" in low
    assert "only by the system" in low
