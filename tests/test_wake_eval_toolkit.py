"""Wake-detection EVALUATION/DEBUG toolkit — the negatives + FA/hour + verifier + spectrogram.

Closes the gap an independent judge flagged: the prior diagnostics measured POSITIVES
only (no negative corpus, no false-accepts/hour, no ROC/DET, no separation metric). This
covers, with mocked models / synthetic clips (no GPU, no openWakeWord wheel needed for the
pure-math paths):

* Task 1 — positives-vs-negatives max-score histogram + the separation scalar.
* Task 2 — FA-EVENT grouping (consecutive frames = ONE event), FA/hour wall-clock math,
  miss-rate at a target FA/h via ``np.interp``, and the empty-corpus graceful message.
* Task 3 — the custom verifier: trains/loads when scikit-learn + openWakeWord are present,
  degrades to a clear message (no crash) when scikit-learn is absent; core imports clean.
* Task 4 — the log-mel spectrogram grid shape + the per-frame score_trace.
* Task 5 — patience/debounce replay threaded into ``score_wake_clip``.
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

import numpy as np
import pytest

from my_stt_tts import __main__ as main_mod
from my_stt_tts import audio
from my_stt_tts.config import Config
from my_stt_tts.events import EventBus
from my_stt_tts.wake import (
    count_fa_events,
    fa_eval,
    fired_with_patience,
    log_mel_spectrogram,
    score_clip_set,
    score_wake_clip,
    separation,
)


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
        return {"maziko": 0.9 if int(np.max(np.abs(frame))) > 1000 else 0.001}


def _drain(sub: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while True:
        try:
            out.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    return out


def _have_scipy() -> bool:
    return importlib.util.find_spec("scipy") is not None


# scipy lives in the installed extras; the log-mel grid degrades to empty without it.
_NEEDS_SCIPY = pytest.mark.skipif(not _have_scipy(), reason="log-mel spectrogram needs scipy")


# =========================================================================== #
# Task 5 — patience / debounce replay in score_wake_clip                       #
# =========================================================================== #
def test_fired_with_patience_single_spike_needs_run() -> None:
    # One above-threshold frame fires at patience=1 (the classic decision)…
    assert fired_with_patience([0.1, 0.9, 0.1], 0.5) is True
    # …but NOT at patience=3 (a one-frame fluke is suppressed).
    assert fired_with_patience([0.1, 0.9, 0.1], 0.5, patience=3) is False
    # Three consecutive above-threshold frames clear patience=3.
    assert fired_with_patience([0.6, 0.7, 0.8], 0.5, patience=3) is True
    # Non-consecutive crossings do NOT count toward the run.
    assert fired_with_patience([0.6, 0.1, 0.6, 0.1, 0.6], 0.5, patience=2) is False


def test_fired_with_patience_empty_and_debounce() -> None:
    assert fired_with_patience([], 0.5) is False
    # debounce alone (patience=1) still fires on the first crossing.
    assert fired_with_patience([0.9, 0.9, 0.9], 0.5, debounce=5) is True


def test_score_wake_clip_patience_gates_a_single_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that spikes for exactly ONE frame fires by default but not under patience>1."""
    scores = iter([0.1, 0.9] + [0.1] * 40)

    class OneSpikeModel:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, _frame: np.ndarray) -> dict[str, float]:
            return {"maziko": next(scores, 0.1)}

    _install_fake_openwakeword(monkeypatch, OneSpikeModel)
    clip = (np.sin(np.linspace(0, 100, 32000)) * 0.4).astype(np.float32)
    # Default (patience=1): the lone 0.9 frame fires.
    conf, fired = score_wake_clip(clip, 16000, "maziko", threshold=0.5, phases=1)
    assert conf == pytest.approx(0.9)
    assert fired is True
    # Replay under ship config (patience=3): the same lone spike does NOT fire,
    # even though confidence (the MAX) is unchanged.
    scores = iter([0.1, 0.9] + [0.1] * 40)
    _install_fake_openwakeword(monkeypatch, OneSpikeModel)
    conf2, fired2 = score_wake_clip(clip, 16000, "maziko", threshold=0.5, phases=1, patience=3)
    assert conf2 == pytest.approx(0.9)
    assert fired2 is False


# =========================================================================== #
# Task 1 — positives-vs-negatives histogram + separation                       #
# =========================================================================== #
def test_score_clip_set_returns_max_scores_and_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_openwakeword(monkeypatch, _EnergyModel)
    loud = (np.sin(np.linspace(0, 100, 16000)) * 0.5).astype(np.float32)
    quiet = np.zeros(16000, dtype=np.float32)
    max_scores, traces = score_clip_set([loud, quiet], "maziko", threshold=0.4, phases=1)
    assert max_scores[0] == pytest.approx(0.9)
    assert max_scores[1] == pytest.approx(0.001)
    # Each clip yields a per-frame trace (used by the FA-eval) — non-empty for ~1 s.
    assert len(traces) == 2
    assert len(traces[0]) >= 5


def test_separation_clean_vs_overlap() -> None:
    clean = separation([0.9, 0.85, 0.8], [0.1, 0.05, 0.15])
    overlap = separation([0.5, 0.45], [0.5, 0.45])
    assert clean > 3.0  # well-separated -> large d-prime
    assert abs(overlap) < 0.1  # indistinguishable -> ~0
    # An empty side is undefined -> 0.0 (not a crash).
    assert separation([0.9], []) == 0.0
    assert separation([], [0.1]) == 0.0


def test_separation_constant_side_degrades_to_mean_gap() -> None:
    # Both sides constant (zero variance) -> d-prime undefined -> report the mean gap.
    assert separation([0.9, 0.9], [0.1, 0.1]) == pytest.approx(0.8)


# =========================================================================== #
# Task 2 — FA-EVENT grouping + FA/hour + miss@target                           #
# =========================================================================== #
def test_count_fa_events_consecutive_frames_are_one_event() -> None:
    # A sustained above-threshold passage is ONE event, not one per frame.
    assert count_fa_events([0.9, 0.9, 0.9, 0.9], 0.5) == 1
    # No crossing -> zero events.
    assert count_fa_events([0.1, 0.2, 0.3], 0.5) == 0
    # Empty trace -> zero.
    assert count_fa_events([], 0.5) == 0


def test_count_fa_events_grouping_window_merges_nearby_crossings() -> None:
    # Two crossings 1 frame apart, within the grouping window -> merged into ONE event.
    assert count_fa_events([0.9, 0.1, 0.9], 0.5, grouping_window=5) == 1
    # Two crossings far apart (gap > window) -> TWO distinct events.
    far = [0.9] + [0.0] * 10 + [0.9]
    assert count_fa_events(far, 0.5, grouping_window=3) == 2


def test_fa_eval_fa_per_hour_and_true_accept_math() -> None:
    # 13 negative frames @ 80 ms = 1.04 s; two well-separated FA events at thr 0.5.
    neg = [[0.9] + [0.0] * 11 + [0.9]]
    # Two positives that both fire at 0.5 -> true_accept 1.0.
    pos = [[0.1, 0.9, 0.1], [0.2, 0.8, 0.1]]
    res = fa_eval(pos, neg, thresholds=[0.5], grouping_window=3, target_fa=100000.0)
    pt = res["points"][0]
    assert pt["threshold"] == 0.5
    assert pt["true_accept"] == 1.0
    # 2 events / 1.04 s * 3600 ~= 6923 FA/hour.
    assert pt["fa_per_hour"] == pytest.approx(2 / (13 * 0.08) * 3600.0, rel=1e-3)
    assert res["neg_seconds"] == pytest.approx(13 * 0.08, rel=1e-3)


def test_fa_eval_miss_at_target_fa_interpolates() -> None:
    # Two thresholds: a loose one (high FA, full recall) and a strict one (zero FA, half
    # recall). The miss-rate at a target FA between them is np.interp-olated.
    pos = [[0.9], [0.3]]  # at thr 0.4 both fire (recall 1.0); at thr 0.6 only one (0.5)
    neg = [[0.5] + [0.0] * 9 + [0.5]]  # at thr 0.4: 2 FA events; at thr 0.6: 0
    res = fa_eval(pos, neg, thresholds=[0.4, 0.6], grouping_window=3, target_fa=0.0)
    # At target_fa=0 -> the strict (zero-FA) point -> miss-rate 0.5 (one of two positives).
    assert res["miss_at_target_fa"] == pytest.approx(0.5)


def test_fa_eval_default_threshold_grid_is_swept() -> None:
    res = fa_eval([[0.5]], [[0.5]], target_fa=0.5)
    assert len(res["points"]) == 19  # np.linspace(0.05, 0.95, 19)
    assert all(0.0 <= p["true_accept"] <= 1.0 for p in res["points"])


# =========================================================================== #
# Task 2 (action) — empty negative corpus emits a clear message, no crash      #
# =========================================================================== #
def test_run_fa_eval_empty_corpus_emits_drop_wavs_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    cfg.negative_corpus_dir = str(tmp_path / "negatives")  # missing dir -> empty corpus
    monkeypatch.setattr("my_stt_tts.wake.wake_model_for", lambda *_a, **_k: __file__)  # "exists"
    bus = EventBus()
    sub = bus.subscribe()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main_mod, "bus", bus)
        main_mod._run_fa_eval(cfg, "maziko")
    result = next(e for e in _drain(sub) if e["type"] == "fa_eval_result")
    assert result["points"] == []
    assert "drop wake-word-free WAVs" in result["message"]
    assert str(tmp_path / "negatives") in result["message"]


def test_run_score_histogram_no_positives_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    monkeypatch.setattr("my_stt_tts.wake.wake_model_for", lambda *_a, **_k: __file__)  # "exists"
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(tmp_path / "rec"))
    bus = EventBus()
    sub = bus.subscribe()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main_mod, "bus", bus)
        main_mod._run_score_histogram(cfg, "maziko")
    result = next(e for e in _drain(sub) if e["type"] == "score_histogram_result")
    assert result["pos_scores"] == []
    assert "no saved positive clips" in result["message"]


def test_run_score_histogram_scores_pos_and_neg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    # Real positives saved on disk + a negative corpus; score_clip_set mocked to a
    # deterministic per-clip score so the histogram math is exercised without a model.
    rec = tmp_path / "rec"
    monkeypatch.setattr(audio, "recordings_dir", lambda: str(rec))
    word_dir = rec / "wake" / "maziko"
    word_dir.mkdir(parents=True)
    neg_dir = tmp_path / "negatives"
    neg_dir.mkdir()
    from my_stt_tts.util import wav_bytes_from_float

    for i in range(3):
        clip = (np.sin(np.linspace(0, 50, 16000)) * 0.4).astype(np.float32)
        (word_dir / f"pos{i}.wav").write_bytes(wav_bytes_from_float(clip, 16000))
    for i in range(2):
        (neg_dir / f"neg{i}.wav").write_bytes(
            wav_bytes_from_float(np.zeros(16000, dtype=np.float32), 16000)
        )
    cfg.negative_corpus_dir = str(neg_dir)
    monkeypatch.setattr("my_stt_tts.wake.wake_model_for", lambda *_a, **_k: __file__)

    def _fake_score_set(clips: list[Any], *_a: Any, **_k: Any) -> tuple[list[float], list[Any]]:
        # positives loud -> 0.8; the all-zero negatives -> 0.05
        out = [0.8 if float(np.max(np.abs(c))) > 0.1 else 0.05 for c in clips]
        return out, [[s] for s in out]

    monkeypatch.setattr("my_stt_tts.wake.score_clip_set", _fake_score_set)
    bus = EventBus()
    sub = bus.subscribe()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main_mod, "bus", bus)
        main_mod._run_score_histogram(cfg, "maziko")
    result = next(e for e in _drain(sub) if e["type"] == "score_histogram_result")
    assert result["pos_scores"] == [0.8, 0.8, 0.8]
    assert result["neg_scores"] == [0.05, 0.05]
    assert result["separation"] > 0.0  # clearly separated


# =========================================================================== #
# Task 4 — log-mel spectrogram grid + score_trace                              #
# =========================================================================== #
@_NEEDS_SCIPY
def test_log_mel_spectrogram_grid_shape() -> None:
    clip = (np.sin(2 * np.pi * 440 * np.arange(32000) / 16000) * 0.4).astype(np.float32)
    spec = log_mel_spectrogram(clip, 16000, n_mels=40)
    assert spec["mels"] == 40
    assert spec["frames"] > 0
    assert len(spec["grid"]) == 40  # one row per mel band
    assert all(len(row) == spec["frames"] for row in spec["grid"])
    assert len(spec["freqs"]) == 40
    assert len(spec["times"]) == spec["frames"]
    # Normalized to [0, 1] for a heatmap.
    assert min(min(r) for r in spec["grid"]) >= 0.0
    assert max(max(r) for r in spec["grid"]) <= 1.0


@_NEEDS_SCIPY
def test_log_mel_spectrogram_downsamples_long_clip() -> None:
    long_clip = (np.random.default_rng(0).standard_normal(16000 * 20) * 0.3).astype(np.float32)
    spec = log_mel_spectrogram(long_clip, 16000, max_frames=200)
    assert spec["frames"] == 200  # capped for the wire


def test_log_mel_spectrogram_empty_clip() -> None:
    spec = log_mel_spectrogram(np.zeros(0, dtype=np.float32), 16000)
    assert spec["frames"] == 0
    assert spec["grid"] == []


def test_log_mel_spectrogram_without_scipy_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    """No scipy -> an empty grid (not a crash), so the worker emits a clear message.

    Forces the scipy import inside ``log_mel_spectrogram`` to fail regardless of whether
    scipy is installed, proving the gated-import degradation path.
    """
    import builtins

    real_import = builtins.__import__

    def _no_scipy(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError("no scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_scipy)
    clip = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 0.4).astype(np.float32)
    spec = log_mel_spectrogram(clip, 16000)
    assert spec["frames"] == 0
    assert spec["grid"] == []


@_NEEDS_SCIPY
def test_run_spectrogram_emits_grid_and_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    clip = (np.sin(2 * np.pi * 300 * np.arange(24000) / 16000) * 0.4).astype(np.float32)
    monkeypatch.setattr(main_mod, "_load_saved_wake_clip", lambda *_a: (clip, "abcd1234", "x.wav"))
    monkeypatch.setattr("my_stt_tts.wake.wake_model_for", lambda *_a, **_k: __file__)
    monkeypatch.setattr(
        "my_stt_tts.wake.score_wake_clip",
        lambda *_a, **_k: (0.5, True, [0.1, 0.5, 0.2]),
    )
    bus = EventBus()
    sub = bus.subscribe()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main_mod, "bus", bus)
        main_mod._run_spectrogram(cfg, "maziko", "abcd1234")
    result = next(e for e in _drain(sub) if e["type"] == "spectrogram_result")
    assert result["hash"] == "abcd1234"
    assert result["mels"] == 40
    assert len(result["grid"]) == 40
    assert result["score_trace"] == [0.1, 0.5, 0.2]


def test_run_spectrogram_no_clip_message(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    monkeypatch.setattr(main_mod, "_load_saved_wake_clip", lambda *_a: (None, "", ""))
    bus = EventBus()
    sub = bus.subscribe()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main_mod, "bus", bus)
        main_mod._run_spectrogram(cfg, "maziko", "deadbeef")
    result = next(e for e in _drain(sub) if e["type"] == "spectrogram_result")
    assert result["grid"] == []
    assert "no saved clip" in result["message"]


# =========================================================================== #
# Task 3 — custom verifier: gated on scikit-learn; core imports clean          #
# =========================================================================== #
def _have_sklearn() -> bool:
    return importlib.util.find_spec("sklearn") is not None


def _have_openwakeword() -> bool:
    return importlib.util.find_spec("openwakeword") is not None


def test_wake_verifier_imports_without_sklearn() -> None:
    """The module imports + its public API is callable even with no scikit-learn / oWW.

    (Core stays clean — the heavy deps are lazily imported inside the functions.)
    """
    import my_stt_tts.wake_verifier as wv

    assert hasattr(wv, "train_verifier")
    assert hasattr(wv, "CustomVerifier")
    # verifier_path is pure (no deps) and traversal-safe.
    assert wv.verifier_path("maziko").endswith("maziko.joblib")
    assert "/" not in Path(wv.verifier_path("../evil")).name


def test_train_verifier_without_sklearn_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With scikit-learn unimportable, training reports trained=False + a clear hint."""
    import builtins

    import my_stt_tts.wake_verifier as wv

    real_import = builtins.__import__

    def _no_sklearn(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "sklearn" or name.startswith("sklearn.") or name == "joblib":
            raise ImportError("no scikit-learn")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_sklearn)
    res = wv.train_verifier([np.zeros(16000)] * 5, [np.zeros(16000)] * 5, "maziko")
    assert res["trained"] is False
    assert "scikit-learn" in res["message"]


def test_train_verifier_too_few_positives(monkeypatch: pytest.MonkeyPatch) -> None:
    """<3 usable positives -> trained=False with the minimum-data message (no crash).

    Mock the embedder so the test does not need the openWakeWord wheel.
    """
    import my_stt_tts.wake_verifier as wv

    if not _have_sklearn():
        pytest.skip("needs scikit-learn (the `debug` extra)")
    monkeypatch.setattr(wv, "_embedding_model", lambda: object())
    monkeypatch.setattr(wv, "embed_clip", lambda *_a, **_k: np.ones(96, dtype=np.float32))
    res = wv.train_verifier([np.zeros(16000)] * 2, [np.zeros(16000)] * 5, "maziko")
    assert res["trained"] is False
    assert "need >=3 positive" in res["message"]


@pytest.mark.skipif(
    not (_have_sklearn() and _have_openwakeword()),
    reason="needs scikit-learn (`debug`) + openWakeWord (`wake`) extras",
)
def test_train_verifier_round_trip(tmp_path: Path) -> None:
    """End-to-end: train on synthetic pos/neg embeddings, save, reload, score (real deps)."""
    import my_stt_tts.wake_verifier as wv

    rng = np.random.default_rng(0)
    pos = [
        (np.sin(2 * np.pi * 300 * np.arange(24000) / 16000) * 0.4).astype(np.float32)
        for _ in range(5)
    ]
    neg = [(rng.standard_normal(24000) * 0.3).astype(np.float32) for _ in range(8)]
    res = wv.train_verifier(pos, neg, "testword", verifier_dir=str(tmp_path))
    assert res["trained"] is True
    assert Path(res["path"]).is_file()
    assert res["n_pos"] == 5
    verifier = wv.CustomVerifier.load("testword", verifier_dir=str(tmp_path))
    assert verifier is not None
    # The verifier discriminates: a positive scores high, a negative low.
    assert verifier.score(pos[0]) > verifier.score(neg[0])


def test_run_train_verifier_action_emits_verifier_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = Config(anthropic_api_key="sk-test")
    monkeypatch.setattr(main_mod, "_load_positive_clips", lambda *_a: [np.zeros(16000)] * 5)
    monkeypatch.setattr(main_mod, "_load_negative_clips", lambda *_a: ([np.zeros(16000)] * 3, "d"))
    monkeypatch.setattr(
        "my_stt_tts.wake_verifier.train_verifier",
        lambda *_a, **_k: {
            "trained": True,
            "path": str(tmp_path / "v.joblib"),
            "n_pos": 5,
            "n_neg": 3,
            "message": "ok",
        },
    )
    bus = EventBus()
    sub = bus.subscribe()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(main_mod, "bus", bus)
        main_mod._run_train_verifier(cfg, "maziko")
    result = next(e for e in _drain(sub) if e["type"] == "verifier_result")
    assert result["trained"] is True
    assert result["n_pos"] == 5
    assert result["n_neg"] == 3


# =========================================================================== #
# WakeWord custom-verifier gate (Task 3, live-loop integration)                #
# =========================================================================== #
def test_wakeword_verifier_gates_base_fire(monkeypatch: pytest.MonkeyPatch) -> None:
    """A base-model fire is GATED: it only counts when the verifier ALSO confirms."""
    from my_stt_tts.wake import WakeWord

    _install_fake_openwakeword(monkeypatch, _EnergyModel)

    class _RejectVerifier:
        def score(self, _clip: np.ndarray, _rate: int = 16000) -> float:
            return 0.1  # below threshold -> rejects the fire

    class _AcceptVerifier:
        def score(self, _clip: np.ndarray, _rate: int = 16000) -> float:
            return 0.9

    loud = (np.ones(1280, dtype=np.float32) * 0.5).astype(np.float32)
    # Without a verifier the loud frame fires.
    bare = WakeWord("wakewords/maziko.onnx", 0.4, phases=1)
    assert any(bare.detect(loud) for _ in range(3)) is True
    # With a REJECTING verifier the same loud audio does NOT fire (gated).
    gated = WakeWord(
        "wakewords/maziko.onnx",
        0.4,
        phases=1,
        custom_verifier=_RejectVerifier(),
        verifier_threshold=0.5,
    )
    assert any(gated.detect(loud) for _ in range(3)) is False
    # With an ACCEPTING verifier it fires again.
    confirmed = WakeWord(
        "wakewords/maziko.onnx",
        0.4,
        phases=1,
        custom_verifier=_AcceptVerifier(),
        verifier_threshold=0.5,
    )
    assert any(confirmed.detect(loud) for _ in range(3)) is True


# =========================================================================== #
# audio.read_wav_float + list_wavs (the reusable loaders)                      #
# =========================================================================== #
def test_read_wav_float_round_trips(tmp_path: Path) -> None:
    from my_stt_tts.util import wav_bytes_from_float

    clip = (np.sin(np.linspace(0, 50, 16000)) * 0.5).astype(np.float32)
    wav = tmp_path / "c.wav"
    wav.write_bytes(wav_bytes_from_float(clip, 16000))
    loaded, rate = audio.read_wav_float(str(wav), target_rate=16000)
    assert rate == 16000
    assert loaded.shape == (16000,)
    np.testing.assert_allclose(loaded, clip, atol=1e-3)


def test_read_wav_float_resamples_48k(tmp_path: Path) -> None:
    from my_stt_tts.util import wav_bytes_from_float

    clip48 = (np.sin(np.linspace(0, 50, 48000)) * 0.4).astype(np.float32)  # 1 s @ 48 kHz
    wav = tmp_path / "c48.wav"
    wav.write_bytes(wav_bytes_from_float(clip48, 48000))
    loaded, rate = audio.read_wav_float(str(wav), target_rate=16000)
    assert rate == 16000
    assert abs(loaded.size - 16000) <= 1  # resampled to ~16 kHz


def test_list_wavs_flat_and_missing(tmp_path: Path) -> None:
    assert audio.list_wavs(str(tmp_path / "nope")) == []  # missing dir -> []
    (tmp_path / "a.wav").write_bytes(b"RIFF")
    (tmp_path / "b.wav").write_bytes(b"RIFF")
    (tmp_path / "c.txt").write_bytes(b"x")
    hits = audio.list_wavs(str(tmp_path))
    assert len(hits) == 2
    assert all(h.endswith(".wav") for h in hits)
