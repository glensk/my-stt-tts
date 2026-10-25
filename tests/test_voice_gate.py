"""Tests for the enrolled-voices gate (fake embeddings; no model, no audio device)."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from my_stt_tts import voice_control as vc
from my_stt_tts import voice_gate as vg


def _wav(path: Path, value: float) -> None:
    """A 1 s clip whose constant sample value lets the fake embedder tell speakers apart."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.Wave_write(str(path)) as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes((np.full(16000, value) * 32767).astype("<i2").tobytes())


def _fake_embed(audio: np.ndarray) -> np.ndarray:
    """Maps a clip's level to a direction: 0.5 -> 'albert', -0.5 -> 'someone else'."""
    level = float(np.mean(audio))
    return np.array([1.0, 0.0]) if level > 0 else np.array([0.0, 1.0])


def test_centroid_drops_an_outlier_clip() -> None:
    vec, kept = vg.centroid([np.array([1.0, 0.0])] * 4 + [np.array([0.0, 1.0])])
    assert kept == 4
    assert vec[0] > 0.99


def test_build_profile_uses_only_that_persons_clips(tmp_path: Path) -> None:
    clips = tmp_path / "wake"
    for i in range(3):
        _wav(clips / "voice_on" / f"2026-enroll_albert-{i}.wav", 0.5)
    _wav(clips / "voice_on" / "2026-enroll_anna-0.wav", -0.5)
    path, used, found = vg.build_profile(
        "albert", embed=_fake_embed, clips_dir=clips, profile_dir=tmp_path / "enroll"
    )
    assert path is not None and (used, found) == (3, 3)
    assert np.load(path)[0] > 0.99
    assert vg.enrolled_speakers(clips) == ["albert", "anna"]


def test_build_profile_needs_three_clips(tmp_path: Path) -> None:
    _wav(tmp_path / "wake" / "voice_on" / "2026-enroll_kid-0.wav", 0.5)
    path, used, found = vg.build_profile(
        "kid", embed=_fake_embed, clips_dir=tmp_path / "wake", profile_dir=tmp_path / "enroll"
    )
    assert path is None and (used, found) == (0, 1)


def test_gate_accepts_enrolled_and_rejects_others(tmp_path: Path) -> None:
    np.save(tmp_path / "albert.npy", np.array([1.0, 0.0]))
    gate = vg.VoiceGate(tmp_path, threshold=0.35, embed=_fake_embed)
    ok, name, sim = gate.check(np.full(16000, 0.5, dtype=np.float32))
    assert ok and name == "albert" and sim > 0.99
    ok, _name, sim = gate.check(np.full(16000, -0.5, dtype=np.float32))
    assert not ok and sim < 0.35
    assert gate.check(np.zeros(100, dtype=np.float32))[0] is False  # too short to judge


def test_gate_without_profiles_lets_everyone_in(tmp_path: Path) -> None:
    gate = vg.VoiceGate(tmp_path, embed=_fake_embed)
    assert not gate.active
    assert gate.check(np.zeros(16000, dtype=np.float32))[0] is True


def test_call_profiles_stay_out_of_the_wake_gate(tmp_path: Path) -> None:
    np.save(tmp_path / "albert.npy", np.array([1.0, 0.0]))
    (tmp_path / "call").mkdir()
    np.save(tmp_path / "call" / "albert.npy", np.array([0.0, 1.0]))
    np.save(tmp_path / "call" / "bob.npy", np.array([0.0, 1.0]))
    gate = vg.VoiceGate(tmp_path, embed=_fake_embed)
    assert set(gate.profiles) == {"albert"}  # the wake gate never sees call/
    assert set(gate.call_profiles) == {"albert", "bob"}
    ok, name, _sim = gate.check(np.full(16000, 0.5, dtype=np.float32))
    assert ok and name == "albert"


def test_score_against_prefers_the_call_profile(tmp_path: Path) -> None:
    np.save(tmp_path / "albert.npy", np.array([1.0, 0.0]))
    np.save(tmp_path / "carol.npy", np.array([1.0, 0.0]))
    (tmp_path / "call").mkdir()
    np.save(tmp_path / "call" / "albert.npy", np.array([0.0, 2.0]))  # normalised on load
    gate = vg.VoiceGate(tmp_path, embed=_fake_embed)
    low = np.full(16000, -0.5, dtype=np.float32)  # fake embedding [0, 1]
    assert gate.score_against(low, "albert") == pytest.approx(1.0)  # call, not the wake one
    assert gate.profile_kind("albert") == "call"
    assert gate.score_against(low, "carol") == pytest.approx(0.0)  # no call profile: wake
    assert gate.profile_kind("carol") == "wake"
    assert gate.score_against(low, "nobody") is None
    assert gate.profile_kind("nobody") is None


def test_call_profile_alone_still_needs_the_model(tmp_path: Path) -> None:
    (tmp_path / "call").mkdir()
    np.save(tmp_path / "call" / "albert.npy", np.array([1.0, 0.0]))
    gate = vg.VoiceGate(tmp_path)  # no embedder yet: preload() would load it
    assert not gate.active  # the wake gate still lets everyone in
    clip = np.full(16000, 0.5, dtype=np.float32)
    assert gate.score_against(clip, "albert", timeout=0.01) is None  # fail closed


class _Det:
    last_score = 0.97
    threshold = 0.75
    model_name = "hey_jarvis"


class _Gate:
    active = True
    threshold = 0.35

    def __init__(self, ok: bool) -> None:
        self.ok = ok

    def check(self, _clip: Any) -> tuple[bool, str, float]:
        return self.ok, "albert", 0.6 if self.ok else 0.1


def test_wake_listener_only_fires_for_accepted_voices() -> None:
    accepted = vc.WakeListener(_Det(), lambda: None, gate=_Gate(True))
    rejected = vc.WakeListener(_Det(), lambda: None, gate=_Gate(False))
    clip = np.zeros(32000, dtype=np.float32)
    assert accepted._accept(clip) is True  # pylint: disable=protected-access
    assert rejected._accept(clip) is False  # pylint: disable=protected-access
    assert vc.WakeListener(_Det(), lambda: None)._accept(None) is True  # pylint: disable=protected-access


# -- fail closed (tp#1235) -------------------------------------------------------------------
def _broken_model() -> Any:
    raise RuntimeError("speechbrain import failed")


def _failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, banners: list[str]
) -> vg.VoiceGate:
    np.save(tmp_path / "albert.npy", np.array([1.0, 0.0]))
    monkeypatch.setattr(vg, "default_embedder", _broken_model)
    gate = vg.VoiceGate(tmp_path, notify=banners.append)
    gate.preload()
    assert gate._ready.wait(5)  # pylint: disable=protected-access
    return gate


def test_model_load_failure_refuses_every_wake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    banners: list[str] = []
    with caplog.at_level("ERROR", logger="my_stt_tts.voice_gate"):
        gate = _failed_gate(tmp_path, monkeypatch, banners)
    assert gate.active and gate.load_failed  # the profiles stay: nobody gets in
    assert gate.check(np.full(16000, 0.5, dtype=np.float32)) == (False, "", 0.0)
    assert gate.unavailable() == "speaker model failed to load"
    assert any("voice check unavailable" in r.getMessage() for r in caplog.records)
    gate.preload()  # a second preload neither reloads nor notifies again
    gate.check(np.full(16000, 0.5, dtype=np.float32))
    assert banners == [vg.UNAVAILABLE_BANNER]
    assert '"' not in vg.UNAVAILABLE_BANNER  # it goes into an AppleScript string


def test_wake_listener_logs_why_a_failed_gate_starts_no_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    gate = _failed_gate(tmp_path, monkeypatch, [])
    listener = vc.WakeListener(_Det(), lambda: None, gate=gate)
    with caplog.at_level("INFO", logger="my_stt_tts.voice_control"):
        assert listener._accept(np.zeros(32000, dtype=np.float32)) is False  # pylint: disable=protected-access
    assert [r.getMessage() for r in caplog.records if r.name == "my_stt_tts.voice_control"] == [
        (
            "🚫 wake refused: hey_jarvis (score 0.97) · voice check unavailable"
            " (speaker model failed to load) — no call"
        )
    ]


def test_gate_refuses_while_the_model_is_not_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    np.save(tmp_path / "albert.npy", np.array([1.0, 0.0]))
    gate = vg.VoiceGate(tmp_path, notify=lambda _text: None)
    ready = gate._ready  # pylint: disable=protected-access
    monkeypatch.setattr(ready, "wait", lambda timeout=None: False)  # still loading after 30 s
    assert gate.check(np.full(16000, 0.5, dtype=np.float32))[0] is False
    assert gate.unavailable() == "speaker model not loaded yet"


def test_gate_refuses_when_the_model_raises_on_a_clip(tmp_path: Path) -> None:
    np.save(tmp_path / "albert.npy", np.array([1.0, 0.0]))

    def crash(_audio: np.ndarray) -> np.ndarray:
        raise RuntimeError("ECAPA crashed")

    gate = vg.VoiceGate(tmp_path, embed=crash)
    assert gate.check(np.full(16000, 0.5, dtype=np.float32)) == (False, "", 0.0)
