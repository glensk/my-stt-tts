"""Threshold / margin calibration, the margin rule and calibrate_voice.py (tp#1235; fakes only)."""
# pylint: disable=missing-function-docstring

from __future__ import annotations

import importlib.util
import json
import logging
import math
import stat
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from my_stt_tts import interference as itf
from my_stt_tts import voice_calibration as vc
from my_stt_tts import voice_gate as vg
from my_stt_tts import voice_profiles as vp
from my_stt_tts.bridge import AUTH_MARGIN_ENV, Authoriser, Transcript
from my_stt_tts.turns import Utterance

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location(
    "calibrate_under_test", _SCRIPTS / "calibrate_voice.py"
)
assert _spec and _spec.loader
cal = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = cal
_spec.loader.exec_module(cal)


def pos(
    score: float, gap: float | None = None, seconds: float = 1.6, short: bool = True
) -> vc.Sample:
    best = None if gap is None else score - gap
    return vc.Sample("enrolled", True, score, best, seconds, short)


def neg(
    score: float, gap: float | None = None, source: str = "synthetic", name: str = ""
) -> vc.Sample:
    best = None if gap is None else score - gap
    return vc.Sample(source, False, score, best, 2.0, True, name or source)


# -- recommend ------------------------------------------------------------------------------
def test_threshold_alone_sits_in_the_middle_of_the_gap() -> None:
    samples = [pos(0.50 + 0.01 * i) for i in range(20)] + [neg(0.30), neg(0.42, name="thorsten")]
    rec = vc.recommend(samples, min_s=1.2)
    assert rec is not None and rec.feasible and rec.margin is None
    # t_lo = 0.42 + 0.02 = 0.44, t_hi = 0.51 (one of 20 may fail) → 0.47 or 0.48
    assert 0.44 <= rec.threshold <= 0.51
    assert rec.outcome.false_accepts == 0 and rec.outcome.short_rate >= 0.95
    assert any("margin rule off" in n for n in rec.notes)


def test_overlap_needs_the_margin_once_enough_others_are_labelled() -> None:
    albert = [pos(0.45 + 0.01 * i, gap=0.20) for i in range(20)]
    others = [neg(0.50, gap=0.02, source="labelled", name=f"o{i}") for i in range(10)]
    rec = vc.recommend(albert + others, min_s=1.2)
    assert rec is not None and rec.margin is not None
    assert rec.threshold <= 0.46  # Albert's 95% first
    assert 0.04 <= rec.margin <= 0.20
    assert rec.feasible and rec.outcome.false_accepts == 0


def test_overlap_without_enough_others_is_reported_not_hidden() -> None:
    albert = [pos(0.45 + 0.01 * i, gap=0.20) for i in range(20)]
    others = [neg(0.50, gap=0.02, source="labelled", name=f"o{i}") for i in range(3)]
    rec = vc.recommend(albert + others, min_s=1.2)
    assert rec is not None and rec.margin is None and not rec.feasible
    assert rec.outcome.false_accepts == 3 and rec.outcome.false_accept_names[0] == "o0"
    assert any("cannot separate" in n for n in rec.notes)
    assert any("label more 'other' clips" in n for n in rec.notes)


def test_margin_cannot_beat_identical_gaps_keeps_albert_first() -> None:
    albert = [pos(0.50 + 0.01 * i, gap=0.05) for i in range(20)]
    others = [neg(0.60, gap=0.10, source="labelled") for _ in range(10)]
    rec = vc.recommend(albert + others, min_s=1.2, min_others=10)
    assert rec is not None and rec.margin is not None
    assert rec.outcome.short_rate >= 0.95 and not rec.feasible


def test_too_few_short_positives_recommend_nothing() -> None:
    samples = [pos(0.6) for _ in range(5)] + [pos(0.6, seconds=1.0) for _ in range(10)]
    assert vc.recommend(samples, min_s=1.2) is None


def test_evaluate_ignores_too_short_positives_and_counts_false_accepts() -> None:
    samples = [pos(0.6), pos(0.1, seconds=0.8), pos(0.7, short=False), neg(0.5), neg(0.2)]
    out = vc.evaluate(samples, 0.4, None, 1.2)
    assert (out.short_accepted, out.short_total, out.accepted, out.total) == (1, 1, 2, 2)
    assert (out.false_accepts, out.negatives) == (1, 2)
    assert math.isnan(vc.Outcome(0.3, None).short_rate)


def test_accepts_applies_the_margin_only_with_negatives() -> None:
    assert vc.accepts(pos(0.5, gap=0.1), 0.4, 0.05)
    assert not vc.accepts(pos(0.5, gap=0.01), 0.4, 0.05)
    assert vc.accepts(pos(0.5), 0.4, 0.05)  # no negatives known: threshold only
    assert not vc.accepts(pos(0.3, gap=0.3), 0.4, None)


# -- the margin rule at runtime ---------------------------------------------------------------
class DetailScorer:
    def __init__(self, score: float, best: float | None) -> None:
        self.score, self.best = score, best
        self.detail_calls = self.plain_calls = 0

    def score_against(self, audio: Any, name: str, *, timeout: float = 5.0) -> float | None:
        del audio, name, timeout
        self.plain_calls += 1
        return self.score

    def score_detail(
        self, audio: Any, name: str, *, timeout: float = 5.0
    ) -> tuple[float, float | None]:
        del audio, name, timeout
        self.detail_calls += 1
        return self.score, self.best


def _t(seconds: float = 1.5) -> Transcript:
    pcm = np.full(int(seconds * 16000), 0.5, dtype=np.float32)
    return Transcript(1, "send that to Peter", 10.0, binding=Utterance(1, 5.0, 5.0 + seconds, pcm))


def test_margin_refuses_a_voice_too_close_to_a_negative(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="my_stt_tts.bridge")
    auth = Authoriser(DetailScorer(0.50, 0.47), "albert", min_utterance_s=1.0, margin=0.05)
    refusal = auth.verify(_t())
    assert refusal is not None and refusal.code == "other_speaker"
    assert "margin 0.03" in caplog.text
    assert (
        Authoriser(DetailScorer(0.50, 0.40), "albert", min_utterance_s=1.0, margin=0.05).verify(
            _t()
        )
        is None
    )


def test_margin_without_negatives_is_threshold_only() -> None:
    scorer = DetailScorer(0.50, None)
    assert Authoriser(scorer, "albert", min_utterance_s=1.0, margin=0.05).verify(_t()) is None
    off = DetailScorer(0.50, 0.49)
    assert Authoriser(off, "albert", min_utterance_s=1.0).verify(_t()) is None
    assert (off.plain_calls, off.detail_calls) == (1, 0)  # no margin: no negatives computed


def test_margin_comes_from_the_environment() -> None:
    assert Authoriser.from_env(None, {AUTH_MARGIN_ENV: "0.07"}).margin == 0.07
    assert Authoriser.from_env(None, {AUTH_MARGIN_ENV: ""}).margin is None
    assert Authoriser.from_env(None, {AUTH_MARGIN_ENV: "junk"}).margin is None
    assert Authoriser.from_env(None, {}).margin is None


def _embed(audio: np.ndarray) -> np.ndarray:
    level = float(np.mean(audio))
    return np.array([math.cos(level * 3), math.sin(level * 3), 0.0])


def test_voice_gate_score_detail_uses_the_negatives_file(tmp_path: Path) -> None:
    np.save(tmp_path / "albert.npy", _embed(np.full(10, 0.5)))
    (tmp_path / "negatives").mkdir()
    np.save(tmp_path / "negatives" / "call.npy", np.stack([_embed(np.full(10, 0.2))]))
    gate = vg.VoiceGate(tmp_path, embed=_embed)
    assert "negatives" not in gate.profiles  # never a person's profile
    detail = gate.score_detail(np.full(8000, 0.5, dtype=np.float32), "albert")
    assert detail is not None and detail[0] == pytest.approx(1.0, abs=1e-5)
    assert detail[1] is not None and detail[1] < 0.9
    assert gate.score_detail(np.full(100, 0.5), "albert") is None  # too short
    assert gate.score_detail(np.full(8000, 0.5), "nobody") is None
    bare = tmp_path / "bare"
    bare.mkdir()
    np.save(bare / "albert.npy", _embed(np.full(10, 0.5)))
    detail = vg.VoiceGate(bare, embed=_embed).score_detail(np.full(8000, 0.5), "albert")
    assert detail is not None and detail[1] is None


# -- calibrate_voice.py ---------------------------------------------------------------------
class FakeVad:
    def is_speech(self, frame: Any) -> bool:
        return bool(np.max(np.abs(frame)) > 0.05)


def _session(root: Path, levels: list[float]) -> None:
    folder = root / "albert" / "20261025-200000"
    takes = []
    for i, level in enumerate(levels, 1):
        name = f"call/{i:02d}-a1.wav"
        vp.write_wav(folder / name, np.full(int(1.6 * 16000), level, dtype=np.float32))
        takes.append({"n": i, "kind": "short", "usable": True, "call_file": name, "lang": "en"})
    vp.write_json(folder / "session.json", {"who": "albert", "takes": takes})


def _enroll(root: Path) -> None:
    _session(root, [0.48 + 0.002 * i for i in range(20)])
    vp.rebuild_profiles("albert", _embed, FakeVad(), root=root, wake_clips=root / "none")


def _deps(lines: list[str]) -> Any:
    return cal.Deps(embedder=lambda: _embed, vad=FakeVad, out=lines.append)


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "-o",
        str(tmp_path / "enroll"),
        "-D",
        str(tmp_path / "clips"),
        "-E",
        str(tmp_path / ".env"),
        *extra,
    ]


def _synthetic(root: Path, levels: list[float]) -> None:
    for i, level in enumerate(levels):
        vp.write_wav(
            root / "negatives" / "synthetic" / f"say-v{i}-1.wav",
            np.full(32000, level, dtype=np.float32),
        )


def test_dry_run_reports_and_writes_nothing(tmp_path: Path) -> None:
    _enroll(tmp_path / "enroll")
    _synthetic(tmp_path / "enroll", [0.1, 0.2])
    (tmp_path / ".env").write_text("OTHER_KEY=keep\n", encoding="utf-8")
    lines: list[str] = []
    assert cal.main(_argv(tmp_path), _deps(lines)) == 0
    text = "\n".join(lines)
    assert "20 enrolled albert" in text and "2 synthetic other" in text
    assert "Recommended: threshold" in text and "dry run" in text
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "OTHER_KEY=keep\n"


def test_apply_writes_threshold_and_margin_with_enough_others(tmp_path: Path) -> None:
    root = tmp_path / "enroll"
    _enroll(root)
    _synthetic(root, [0.1, 0.2])
    store = itf.ClipStore(tmp_path / "clips")
    for i in range(10):
        clip = store.save(
            np.full(24000, 0.30 + 0.001 * i, dtype=np.float32),
            decision="skipped",
            score=None,
            profile=None,
        )
        store.set_label(clip, "other")
    env = tmp_path / ".env"
    env.write_text("OTHER_KEY=keep\nMAC_VOICE_AUTH_THRESHOLD=0.35\n", encoding="utf-8")
    env.chmod(0o600)
    lines: list[str] = []
    assert cal.main(_argv(tmp_path, "-a"), _deps(lines)) == 0
    written = env.read_text(encoding="utf-8")
    assert "OTHER_KEY=keep" in written and "MAC_VOICE_AUTH_THRESHOLD=0.35" not in written
    assert "MAC_VOICE_AUTH_MARGIN=" in written
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    negatives = root / "negatives" / "call.npy"
    assert np.load(negatives).shape == (12, 3)
    assert stat.S_IMODE(negatives.stat().st_mode) == 0o600
    assert "restart mac-voice" in lines[-1]


def test_apply_without_enough_others_removes_an_old_margin(tmp_path: Path) -> None:
    _enroll(tmp_path / "enroll")
    env = tmp_path / ".env"
    env.write_text("MAC_VOICE_AUTH_MARGIN=0.05\n", encoding="utf-8")
    lines: list[str] = []
    assert cal.main(_argv(tmp_path, "-a"), _deps(lines)) == 0
    written = env.read_text(encoding="utf-8")
    assert "MAC_VOICE_AUTH_MARGIN" not in written and "MAC_VOICE_AUTH_THRESHOLD=" in written


def test_apply_is_refused_until_enroll_voice_built_the_profile(tmp_path: Path) -> None:
    root = tmp_path / "enroll"
    (root / "call").mkdir(parents=True)
    np.save(root / "call" / "albert.npy", _embed(np.full(10, 0.5)))
    store = itf.ClipStore(tmp_path / "clips")
    for i in range(10):
        clip = store.save(
            np.full(32000, 0.5 + 0.001 * i, dtype=np.float32),
            decision="verified",
            score=None,
            profile=None,
        )
        store.set_label(clip, "albert")
    env = tmp_path / ".env"
    env.write_text("MAC_VOICE_AUTH_THRESHOLD=0.35\n", encoding="utf-8")
    lines: list[str] = []
    assert cal.main(_argv(tmp_path, "-a"), _deps(lines)) == 1
    assert "no enrollment yet" in "\n".join(lines)
    assert env.read_text(encoding="utf-8") == "MAC_VOICE_AUTH_THRESHOLD=0.35\n"


def test_a_changed_profile_is_not_trusted(tmp_path: Path) -> None:
    root = tmp_path / "enroll"
    _enroll(root)
    np.save(root / "call" / "albert.npy", np.array([0.0, 0.0, 1.0]))
    lines: list[str] = []
    assert cal.main(_argv(tmp_path), _deps(lines)) == 1  # no positives left → nothing
    assert "changed since enroll_voice.py built it" in "\n".join(lines)


def test_no_call_profile_is_a_usage_error(tmp_path: Path) -> None:
    lines: list[str] = []
    assert cal.main(_argv(tmp_path), _deps(lines)) == 2


def test_negatives_never_compare_with_themselves() -> None:
    clips = [
        cal.Clip("synthetic", False, np.array([1.0, 0.0]), 2.0, True, "a"),
        cal.Clip("synthetic", False, np.array([0.0, 1.0]), 2.0, True, "b"),
        cal.Clip("enrolled", True, np.array([1.0, 0.0]), 2.0, True, "p", 0.9),
    ]
    samples, matrix = cal.to_samples(clips)
    assert matrix is not None and matrix.shape == (2, 2)
    assert samples[0].best_negative == pytest.approx(0.0)  # only "b" to compare with
    assert samples[2].best_negative == pytest.approx(1.0)


def test_say_voices_keeps_real_voices_once() -> None:
    listing = (
        "Albert              en_US    # Hello! My name is Albert.\n"
        "Anna (German (Germany)) de_DE    # Hallo! Ich heiße Anna.\n"
        "Eddy (English (UK)) en_GB    # Hello! My name is Eddy.\n"
        "Eddy (German (Germany)) de_DE    # Hallo! Ich heiße Eddy.\n"
        "Kyoko               ja_JP    # こんにちは\n"
        "Fred                en_US    # Hello! My name is Fred.\n"
    )
    assert cal.say_voices(listing) == [
        ("Anna (German (Germany))", "de"),
        ("Eddy (German (Germany))", "de"),
        ("Fred", "en"),
    ]


def test_generate_synthetic_replaces_old_clips(tmp_path: Path) -> None:
    folder = tmp_path / "synthetic"
    folder.mkdir()
    (folder / "say-old-1.wav").write_bytes(b"x")
    calls: list[list[str]] = []

    def run(cmd: list[str], stdin: str | None = None) -> bool:
        del stdin
        calls.append(cmd)
        if cmd[0].endswith("afconvert"):
            vp.write_wav(Path(cmd[-1]), np.zeros(1600))
        return True

    deps = cal.Deps(
        run=run,
        say_list=lambda: "Anna (German (Germany)) de_DE    # Hallo",
        piper=lambda: "/bin/piper",
        voices_dir=tmp_path / "voices",
    )
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "de_DE-thorsten-high.onnx").write_bytes(b"")
    assert cal.generate_synthetic(folder, deps) == 4
    assert sorted(p.name for p in folder.iterdir()) == [
        "piper-de_DE-thorsten-high-1.wav",
        "piper-de_DE-thorsten-high-2.wav",
        "say-anna-german-germany-1.wav",
        "say-anna-german-germany-2.wav",
    ]
    assert any(c[0] == "/usr/bin/say" and "Anna (German (Germany))" in c for c in calls)


def test_manifest_records_the_profile_hash(tmp_path: Path) -> None:
    _enroll(tmp_path)
    manifest = json.loads((tmp_path / "albert" / "profiles.json").read_text(encoding="utf-8"))
    assert manifest["call"]["sha256"] == vp.sha256_file(tmp_path / "call" / "albert.npy")
