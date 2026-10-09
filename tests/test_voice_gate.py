"""Tests for the enrolled-voices gate (fake embeddings; no model, no audio device)."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Any

import numpy as np

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
