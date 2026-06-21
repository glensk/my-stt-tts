"""Mycroft Precise ideas, ported (precise-ideas branch, repo #5 of the wake checker loop).

TWO portable wins from Precise (the IDEAS, NOT its GRU):

Feature 1 — OUTPUT CALIBRATION (Precise's ``ThresholdDecoder``): a per-word monotone map
``Φ((logit(raw) − μ)/σ)`` fit from the saved positive-clip score stats, applied to the wake
score so ``threshold=0.5`` is model-independent. Covered here:

* the map is MONOTONE increasing, in ``[0, 1]``, identity when OFF / insufficient samples;
* it is applied IDENTICALLY in :meth:`WakeWord.detect` (live) and :func:`score_wake_clip`
  (eval) — the LIVE == EVAL invariant — via a fake openWakeWord model;
* the ``wake_calibration`` config knob (default / env / settings_dict / apply_settings /
  settings_text) + persistence round-trip.

Feature 2 — ACTIVE-LEARNING CLOSED LOOP (port the LOOP, keep our cheap CPU rebuilders):

* ``save_recording(kind="wake_neg")`` lands in ``debug/recordings/wake_neg/<word>/``, and
  ``_load_negative_clips(cfg, word)`` UNIONS it with ``negative_corpus_dir``;
* ``mark_false_fire`` / ``mark_miss`` move the clip to the right per-word dir and rebuild;
* the EVAL-GATED rebuild ACCEPTS an improving rebuild and ROLLS BACK a regressing one
  (the golden-enrollment safety interlock), restoring the prior model artifacts byte-for-byte;
* ``capture_last_fire`` saves the ring-buffer audio as a negative then relabels;
* the ring buffer (:class:`WakeFireBuffer`) retains + snapshots the last fire's audio;
* ``record_wake_outcome`` stores the clip hash/path so a logged outcome is actionable;
* the ``relabel_result`` event shape.

openWakeWord / scikit-learn are mocked; clips/disk use tmp dirs.
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring,import-outside-toplevel,redefined-outer-name

from __future__ import annotations

import json
import sys
import threading
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from my_stt_tts.calibration import Calibrator, calibrator_for, fit_and_save
from my_stt_tts.config import Config
from my_stt_tts.events import EventBus
from my_stt_tts.wake import WakeWord, score_wake_clip


# --------------------------------------------------------------------------- #
# Fake openWakeWord plumbing (shared with the other wake tests' pattern)      #
# --------------------------------------------------------------------------- #
def _install_fake_openwakeword(monkeypatch: pytest.MonkeyPatch, model_cls: type) -> None:
    pkg = types.ModuleType("openwakeword")
    mod = types.ModuleType("openwakeword.model")
    mod.Model = model_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.model", mod)


class _FixedModel:
    """A fake oWW model that always returns the SAME raw score (so calibration is testable)."""

    _score: float = 0.6

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def predict(self, _frame: np.ndarray) -> dict[str, float]:
        return {"w": type(self)._score}

    def reset(self) -> None:
        return None


def _make_fixed(monkeypatch: pytest.MonkeyPatch, score: float) -> type:
    cls = type("Fixed", (_FixedModel,), {"_score": float(score)})
    _install_fake_openwakeword(monkeypatch, cls)
    return cls


# =========================================================================== #
# FEATURE 1 — OUTPUT CALIBRATION                                              #
# =========================================================================== #
def test_calibrator_identity_when_off_or_insufficient() -> None:
    ident = Calibrator.identity()
    assert not ident.enabled
    assert ident.apply(0.3) == 0.3
    assert ident.apply(0.99) == 0.99
    # too few positive samples -> identity (calibration silently OFF)
    cal = Calibrator.fit([0.5, 0.6])
    assert not cal.enabled
    assert cal.apply(0.42) == 0.42


def test_calibrator_map_is_monotone_and_bounded() -> None:
    pos = [0.6, 0.65, 0.7, 0.62, 0.68, 0.66, 0.71, 0.64]
    cal = Calibrator.fit(pos)
    assert cal.enabled
    xs = np.linspace(0.001, 0.999, 60)
    ys = [cal.apply(float(x)) for x in xs]
    # monotone non-decreasing
    assert all(ys[i + 1] >= ys[i] - 1e-12 for i in range(len(ys) - 1))
    # bounded to [0, 1]
    assert all(0.0 <= y <= 1.0 for y in ys)
    # 0.5 lands at the LOW edge of the positive cluster: a median positive calibrates ABOVE 0.5
    assert cal.apply(float(np.median(pos))) > 0.5


def test_calibrator_apply_scalar_and_array_agree() -> None:
    cal = Calibrator(mu=0.0, sigma=0.5, n=10)
    arr = cal.apply(np.array([0.2, 0.5, 0.8]))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (3,)
    for i, raw in enumerate([0.2, 0.5, 0.8]):
        assert abs(float(arr[i]) - cal.apply(raw)) < 1e-9


def test_calibrator_persist_round_trip(tmp_path: Path) -> None:
    cal = Calibrator(mu=0.1, sigma=0.4, n=7)
    out = cal.save("maziko", calibration_dir=str(tmp_path))
    assert out and Path(out).is_file()
    loaded = Calibrator.load("maziko", calibration_dir=str(tmp_path))
    assert loaded.enabled
    assert loaded.mu is not None and loaded.sigma is not None
    assert abs(loaded.mu - 0.1) < 1e-9 and abs(loaded.sigma - 0.4) < 1e-9 and loaded.n == 7
    # identity calibrator writes nothing; load of a missing file is identity
    assert Calibrator.identity().save("none", calibration_dir=str(tmp_path)) == ""
    assert not Calibrator.load("never_fit", calibration_dir=str(tmp_path)).enabled


def test_calibrator_for_respects_switch(tmp_path: Path) -> None:
    fit_and_save("maziko", [0.6, 0.65, 0.7, 0.62, 0.68, 0.66], calibration_dir=str(tmp_path))
    # switch OFF -> always identity (no disk read needed)
    assert not calibrator_for("maziko", enabled=False, calibration_dir=str(tmp_path)).enabled
    # switch ON + a persisted fit -> real map
    assert calibrator_for("maziko", enabled=True, calibration_dir=str(tmp_path)).enabled
    # switch ON but no fit -> identity
    assert not calibrator_for("absent", enabled=True, calibration_dir=str(tmp_path)).enabled


def test_calibration_applied_in_detect_live(monkeypatch: pytest.MonkeyPatch) -> None:
    _make_fixed(monkeypatch, 0.6)
    cal = Calibrator(mu=0.0, sigma=0.5, n=10)
    det = WakeWord("wakewords/w.onnx", threshold=0.5, phases=1, calibrator=cal)
    frame = np.full(1280, 0.1, dtype=np.float32)
    det.detect(frame)
    # last_score is the CALIBRATED value of the raw 0.6, not 0.6 itself
    assert abs(det.last_score - cal.apply(0.6)) < 1e-9
    assert det.last_score != pytest.approx(0.6)


def test_calibration_identity_detect_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    _make_fixed(monkeypatch, 0.6)
    # No calibrator (None) -> identity -> raw score preserved exactly.
    det = WakeWord("wakewords/w.onnx", threshold=0.5, phases=1)
    det.detect(np.full(1280, 0.1, dtype=np.float32))
    assert det.last_score == pytest.approx(0.6)


def test_calibration_live_equals_eval(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The map is applied IDENTICALLY in detect (live) and score_wake_clip (eval)."""
    _make_fixed(monkeypatch, 0.6)
    # A model file must exist for score_wake_clip; point wake_model_for at a tmp .onnx.
    model = tmp_path / "w.onnx"
    model.write_bytes(b"x")
    cal = Calibrator(mu=0.0, sigma=0.5, n=10)

    # LIVE: drive detect frame-by-frame, collect last_score per frame.
    det = WakeWord(str(model), threshold=0.5, phases=1, calibrator=cal)
    clip = np.full(1280 * 6, 0.1, dtype=np.float32)
    live_trace = []
    for start in range(0, clip.size, 1280):
        det.detect(clip[start : start + 1280])
        live_trace.append(round(det.last_score, 4))

    # EVAL: same clip, same calibrator, through score_wake_clip with the trace.
    conf, _fired, eval_trace = score_wake_clip(
        clip,
        16000,
        "w",
        threshold=0.5,
        phases=1,
        wakewords_dir=str(tmp_path),
        calibrator=cal,
        with_trace=True,
    )
    # Every frame's calibrated score matches between the live detector and the eval path.
    assert eval_trace == live_trace
    # And confidence is the calibrated value (not the raw 0.6).
    assert conf == pytest.approx(cal.apply(0.6), abs=1e-4)


def test_wake_calibration_config_default_env_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts.webui import apply_settings, settings_dict

    # default OFF
    cfg = Config(anthropic_api_key="x")
    assert cfg.wake_calibration is False
    # env ON
    monkeypatch.setenv("WAKE_CALIBRATION", "true")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")  # so validate() doesn't fail on the key
    cfg_on = Config.from_env()
    assert cfg_on.wake_calibration is True
    cfg_on.validate()  # must not raise (the calibration knob is a bool, no bound)
    # settings_dict surfaces it
    sd = settings_dict(cfg)
    assert sd["wake_calibration"] is False
    # apply_settings toggles it
    apply_settings(cfg, {"wake_calibration": True})
    assert cfg.wake_calibration is True
    # settings_text mentions calibration
    assert "calibration" in main_mod.settings_text(cfg, color=False)


# =========================================================================== #
# FEATURE 2 — ACTIVE-LEARNING CLOSED LOOP                                     #
# =========================================================================== #
def _write_wav(path: Path, clip: np.ndarray, rate: int = 16000) -> None:
    from my_stt_tts.util import wav_bytes_from_float

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes_from_float(np.asarray(clip, dtype=np.float32), rate))


def test_save_recording_wake_neg_per_word_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from my_stt_tts import audio

    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path))
    clip = (np.sin(np.linspace(0, 50, 16000)) * 0.3).astype(np.float32)
    path, hash8, url = audio.save_recording(
        clip, 16000, kind="wake_neg", source="server", word="maziko"
    )
    assert Path(path).is_file()
    assert "wake_neg/maziko/" in url
    assert Path(path).parent == tmp_path / "wake_neg" / "maziko"
    assert hash8 and hash8 in Path(path).name


def test_load_negative_clips_unions_wake_neg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from my_stt_tts import __main__ as main_mod
    from my_stt_tts import active_learning, audio

    rec = tmp_path / "recordings"
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(rec))
    neg_corpus = tmp_path / "negatives"
    # one clip in the shared corpus, one in the per-word wake_neg folder
    _write_wav(neg_corpus / "room.wav", np.full(8000, 0.1, dtype=np.float32))
    _write_wav(
        Path(active_learning.neg_dir_for("maziko")) / "x.wav", np.full(8000, 0.2, dtype=np.float32)
    )
    cfg = Config(anthropic_api_key="x", negative_corpus_dir=str(neg_corpus))
    # word=None -> shared corpus only (back-compat)
    clips_shared, _d = main_mod._load_negative_clips(cfg)
    assert len(clips_shared) == 1
    # word given -> UNION (shared + wake_neg)
    clips_union, primary = main_mod._load_negative_clips(cfg, "maziko")
    assert len(clips_union) == 2
    assert primary == str(neg_corpus)


def test_record_wake_outcome_stores_hash_path(tmp_path: Path) -> None:
    from my_stt_tts.config import load_wake_stats, record_wake_outcome

    stats = tmp_path / "wake_stats.json"
    record_wake_outcome(
        "maziko",
        confidence=0.6,
        fired=True,
        source="server",
        path=str(stats),
        clip_hash="deadbeef",
        clip_path="/x/y-deadbeef.wav",
    )
    loaded = load_wake_stats(str(stats))
    entry = loaded["maziko"][-1]
    assert entry["hash"] == "deadbeef"
    assert entry["clip_path"] == "/x/y-deadbeef.wav"
    # legacy call (no clip) omits the fields
    record_wake_outcome("maziko", confidence=0.1, fired=False, source="server", path=str(stats))
    assert "hash" not in load_wake_stats(str(stats))["maziko"][-1]


def test_wake_fire_buffer_retains_and_snapshots() -> None:
    from my_stt_tts.audio import WakeFireBuffer

    buf = WakeFireBuffer(sample_rate=100, window_seconds=1.0)  # 100-sample window
    assert buf.last_fire is None
    buf.feed(np.arange(60, dtype=np.float32))
    buf.feed(np.arange(60, 120, dtype=np.float32))  # total 120 > 100 -> trims to last 100
    buf.mark_fire()
    snap = buf.last_fire
    assert snap is not None and snap.size == 100
    assert snap[-1] == pytest.approx(119.0)  # most-recent sample retained
    # reset clears the rolling buffer but keeps the snapshot
    buf.reset()
    buf.mark_fire()
    assert buf.last_fire is not None and buf.last_fire.size == 0


def test_gate_improves_decision() -> None:
    from my_stt_tts.active_learning import Gate, gate_improves

    before = Gate(separation=1.0, miss_at_target_fa=0.40)
    # improvement on both axes -> accept
    assert gate_improves(before, Gate(separation=1.5, miss_at_target_fa=0.20))
    # flat (within tolerance) -> accept
    assert gate_improves(before, Gate(separation=1.0, miss_at_target_fa=0.40))
    # separation regresses -> reject
    assert not gate_improves(before, Gate(separation=0.5, miss_at_target_fa=0.40))
    # miss-rate rises -> reject
    assert not gate_improves(before, Gate(separation=1.0, miss_at_target_fa=0.60))


# --------------------------------------------------------------------------- #
# EVAL-GATED rebuild: ACCEPT an improving rebuild, ROLL BACK a regressing one. #
# The rebuilders + gate are stubbed so the test drives the interlock logic     #
# deterministically (the real rebuilders need openWakeWord + scikit-learn).    #
# --------------------------------------------------------------------------- #
def _setup_relabel(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Config]:
    from my_stt_tts import active_learning, audio

    rec = tmp_path / "recordings"
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(rec))
    cfg = Config(anthropic_api_key="x", negative_corpus_dir=str(tmp_path / "negatives"))
    return active_learning, cfg


def test_rebuild_accept_keeps_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    al, cfg = _setup_relabel(monkeypatch, tmp_path)
    Gate = al.Gate

    # before gate: d-prime 1.0 / miss 0.4; after: improved (1.5 / 0.2)
    gates = iter([Gate(1.0, 0.40), Gate(1.5, 0.20)])
    monkeypatch.setattr(al, "compute_gate", lambda *a, **k: next(gates))
    monkeypatch.setattr(al, "_rebuild_detector", lambda *a, **k: True)
    monkeypatch.setattr(al, "load_positive_clips", lambda *a, **k: [])
    monkeypatch.setattr(al, "load_negative_clips_union", lambda *a, **k: ([], []))
    # track whether rollback (restore) was called
    restored = {"called": False}
    monkeypatch.setattr(
        al, "_restore_artifacts", lambda *a, **k: restored.__setitem__("called", True)
    )
    monkeypatch.setattr(al, "_snapshot_artifacts", lambda *a, **k: {})

    res = al.rebuild_and_gate("maziko", cfg, action="mark_false_fire", clip_hash="h")
    assert res.rebuilt and res.accepted
    assert not restored["called"]  # accepted -> NO rollback
    assert res.sep_before == 1.0 and res.sep_after == 1.5
    assert res.fa_before == 0.40 and res.fa_after == 0.20


def test_rebuild_regress_rolls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    al, cfg = _setup_relabel(monkeypatch, tmp_path)
    Gate = al.Gate

    # before gate: d-prime 1.0 / miss 0.4; after: WORSE (0.5 / 0.6) -> must roll back
    gates = iter([Gate(1.0, 0.40), Gate(0.5, 0.60)])
    monkeypatch.setattr(al, "compute_gate", lambda *a, **k: next(gates))
    monkeypatch.setattr(al, "_rebuild_detector", lambda *a, **k: True)
    monkeypatch.setattr(al, "load_positive_clips", lambda *a, **k: [])
    monkeypatch.setattr(al, "load_negative_clips_union", lambda *a, **k: ([], []))
    restored = {"called": False}
    monkeypatch.setattr(
        al, "_restore_artifacts", lambda *a, **k: restored.__setitem__("called", True)
    )
    monkeypatch.setattr(al, "_snapshot_artifacts", lambda *a, **k: {"embeddings": b"golden"})

    res = al.rebuild_and_gate("maziko", cfg, action="mark_false_fire", clip_hash="h")
    assert res.rebuilt and not res.accepted
    assert restored["called"]  # regressed -> ROLLED BACK
    # the reported AFTER equals BEFORE (the detector is unchanged after rollback)
    assert res.sep_after == res.sep_before == 1.0
    assert res.fa_after == res.fa_before == 0.40


def test_snapshot_restore_round_trip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The rollback restores the model artifacts byte-for-byte (golden enrollment floor)."""
    from my_stt_tts import active_learning as al

    # Point the artifact path at tmp by monkeypatching the dirs via the path helpers' modules.
    emb = tmp_path / "wake_embeddings" / "maziko.npz"
    monkeypatch.setattr(al, "_artifact_paths", lambda word: {"embeddings": str(emb)})
    emb.parent.mkdir(parents=True)
    emb.write_bytes(b"GOLDEN")
    snap = al._snapshot_artifacts("maziko")
    assert snap["embeddings"] == b"GOLDEN"
    # a "rebuild" overwrites it...
    emb.write_bytes(b"POISONED")
    # ...rollback restores the golden bytes
    al._restore_artifacts("maziko", snap)
    assert emb.read_bytes() == b"GOLDEN"
    # an artifact ABSENT at snapshot time is removed on rollback (the rebuild created it)
    emb.unlink()
    snap2 = al._snapshot_artifacts("maziko")
    assert snap2["embeddings"] is None
    emb.write_bytes(b"NEW")
    al._restore_artifacts("maziko", snap2)
    assert not emb.exists()


def test_relabel_clip_moves_to_negatives(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from my_stt_tts import active_learning as al
    from my_stt_tts import audio

    rec = tmp_path / "recordings"
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(rec))
    cfg = Config(anthropic_api_key="x")
    # a saved positive clip addressed by hash
    src = rec / "wake" / "maziko" / "20260101-000000-server-abc12345.wav"
    _write_wav(src, np.full(8000, 0.2, dtype=np.float32))
    # stub the rebuild so we only test the MOVE
    monkeypatch.setattr(
        al,
        "rebuild_and_gate",
        lambda w, c, action, clip_hash: al.RelabelResult(
            w, action, True, True, 1.0, 1.2, 0.3, 0.2, "ok", clip_hash
        ),
    )
    res = al.relabel_clip("maziko", "abc12345", "mark_false_fire", cfg)
    assert res.rebuilt and res.accepted
    # the clip MOVED from wake/ to wake_neg/
    assert not src.exists()
    moved = Path(al.neg_dir_for("maziko")) / src.name
    assert moved.is_file()


def test_relabel_clip_mark_miss_moves_to_positives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from my_stt_tts import active_learning as al
    from my_stt_tts import audio

    rec = tmp_path / "recordings"
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(rec))
    cfg = Config(anthropic_api_key="x")
    src = rec / "wake_neg" / "maziko" / "20260101-000000-server-def67890.wav"
    _write_wav(src, np.full(8000, 0.2, dtype=np.float32))
    monkeypatch.setattr(
        al,
        "rebuild_and_gate",
        lambda w, c, action, clip_hash: al.RelabelResult(
            w, action, True, True, 1.0, 1.1, 0.3, 0.25, "ok", clip_hash
        ),
    )
    res = al.relabel_clip("maziko", "def67890", "mark_miss", cfg)
    assert res.action == "mark_miss"
    assert not src.exists()
    assert (Path(al.pos_dir_for("maziko")) / src.name).is_file()


def test_relabel_clip_missing_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from my_stt_tts import active_learning as al
    from my_stt_tts import audio

    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path / "recordings"))
    cfg = Config(anthropic_api_key="x")
    res = al.relabel_clip("maziko", "nope0000", "mark_false_fire", cfg)
    assert not res.rebuilt and not res.accepted
    assert "no saved clip" in res.message


def test_capture_last_fire_saves_negative_then_relabels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from my_stt_tts import active_learning as al
    from my_stt_tts import audio

    rec = tmp_path / "recordings"
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(rec))
    cfg = Config(anthropic_api_key="x")
    seen = {}
    monkeypatch.setattr(
        al,
        "rebuild_and_gate",
        lambda w, c, action, clip_hash: (
            seen.update({"action": action, "hash": clip_hash})
            or al.RelabelResult(w, action, True, True, 1.0, 1.0, 0.3, 0.3, "ok", clip_hash)
        ),
    )
    clip = (np.sin(np.linspace(0, 50, 16000)) * 0.3).astype(np.float32)
    res = al.capture_last_fire("maziko", clip, 16000, cfg)
    assert res.action == "capture_last_fire"
    # the audio was saved as a wake_neg clip for the word (the file exists under wake_neg/)
    saved = list((rec / "wake_neg" / "maziko").glob("*.wav"))
    assert len(saved) == 1
    assert seen["action"] == "capture_last_fire" and seen["hash"]
    # empty clip -> no rebuild
    empty = al.capture_last_fire("maziko", np.zeros(0, dtype=np.float32), 16000, cfg)
    assert not empty.rebuilt


class _FakeStream:
    """Drives ``_callback`` with scripted device blocks then idles (shared listen_for_wake stub)."""

    def __init__(self, blocks: list[np.ndarray], callback: Any) -> None:
        self._blocks = blocks
        self._cb = callback

    def __enter__(self) -> _FakeStream:
        for block in self._blocks:
            self._cb(block.reshape(-1, 1), len(block), None, None)
        return self

    def __exit__(self, *_exc: object) -> None:
        return None


def test_listen_for_wake_fire_buffer_captures_on_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live loop feeds the ring buffer every frame and SNAPSHOTS it on a fire."""
    from my_stt_tts import audio

    # Two frames; the detector fires on the SECOND, so the snapshot must hold the fed audio.
    blocks = [np.full(1280, 0.3, dtype=np.float32), np.full(1280, 0.4, dtype=np.float32)]

    class FiringOnSecond:
        threshold = 0.5
        model_name = "maziko"
        last_score = 0.9

        def __init__(self) -> None:
            self._n = 0

        def reset(self) -> None:
            self._n = 0

        def detect(self, _frame: np.ndarray) -> bool:
            self._n += 1
            return self._n >= 2  # fire on the second frame

    buf = audio.WakeFireBuffer(sample_rate=16000, window_seconds=2.0)
    sd = MagicMock()
    sd.InputStream.side_effect = lambda **kw: _FakeStream(blocks, kw["callback"])
    monkeypatch.setattr(audio, "_sd", lambda: sd)
    monkeypatch.setattr(audio, "_supported_capture_rate", lambda _sd, r: r)

    fired = audio.listen_for_wake(
        FiringOnSecond(), 16000, poll_seconds=0.01, stop=threading.Event(), fire_buffer=buf
    )
    assert fired is True
    snap = buf.last_fire
    assert snap is not None
    # The snapshot holds both fed frames (the audio that led up to + triggered the fire).
    assert snap.size == 2560
    assert snap[0] == pytest.approx(0.3) and snap[-1] == pytest.approx(0.4)


def test_relabel_result_event_shape() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    bus.relabel_result(
        word="maziko",
        action="mark_false_fire",
        rebuilt=True,
        accepted=False,
        sep_before=1.0,
        sep_after=0.5,
        fa_before=0.3,
        fa_after=0.6,
        message="rolled back",
        hash="abc12345",
    )
    evt = json.loads(sub.get_nowait())
    assert evt["type"] == "relabel_result"
    assert evt["word"] == "maziko"
    assert evt["action"] == "mark_false_fire"
    assert evt["rebuilt"] is True
    assert evt["accepted"] is False
    assert evt["sep_before"] == 1.0 and evt["sep_after"] == 0.5
    assert evt["fa_before"] == 0.3 and evt["fa_after"] == 0.6
    assert evt["message"] == "rolled back"
    assert evt["hash"] == "abc12345"
