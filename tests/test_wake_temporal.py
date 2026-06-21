"""Live temporal-smoothing wake criterion (wake-temporal branch, repo #3 microWakeWord).

Ports microWakeWord's runtime idea into the LIVE path: a sliding-window moving-average
fire criterion (mean of the last ``wake_window`` per-frame scores >= threshold) plus a
post-fire ``wake_refractory`` lockout. Covers, with openWakeWord mocked:

* the moving-average fire (mean >= threshold), and that ``window == 1`` is byte-identical
  to the old single-frame ``last_score >= threshold`` decision;
* a one-frame DIP below threshold still fires under a window that averages it out, and a
  one-frame SPIKE alone does NOT fire under a window > 1;
* the refractory lockout suppresses re-fires for N frames after a fire;
* ``reset()`` clears BOTH the moving-average window and the refractory;
* eval-path consistency: ``score_wake_clip`` uses the SAME criterion (live == eval), and
  the pure helpers ``count_fires_moving_average`` / ``moving_average_fires`` replay it;
* the ``wake_window`` / ``wake_refractory`` config knobs (default / env / validate /
  settings_dict / apply_settings / settings_text).
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring

from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest

from my_stt_tts.wake import (
    WakeWord,
    count_fires_moving_average,
    moving_average_fires,
)


def _install_fake_openwakeword(monkeypatch: pytest.MonkeyPatch, model_cls: type) -> None:
    pkg = types.ModuleType("openwakeword")
    mod = types.ModuleType("openwakeword.model")
    mod.Model = model_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.model", mod)


class _ScriptedModel:
    """A fake oWW model that returns a scripted score per predict() call, in order.

    Used with ``phases=1`` so each 1280-sample frame fed to ``detect`` yields exactly one
    predict() and thus one scripted score — the per-frame trace under test.
    """

    _scores: list[float] = []
    _idx = 0

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def predict(self, _frame: np.ndarray) -> dict[str, float]:
        cls = type(self)
        score = cls._scores[cls._idx] if cls._idx < len(cls._scores) else 0.0
        cls._idx += 1
        return {"w": score}

    def reset(self) -> None:
        return None


def _make_scripted(monkeypatch: pytest.MonkeyPatch, scores: list[float]) -> type:
    cls = type("Scripted", (_ScriptedModel,), {"_scores": list(scores), "_idx": 0})
    _install_fake_openwakeword(monkeypatch, cls)
    return cls


def _drive(det: WakeWord, n_frames: int) -> list[bool]:
    """Feed ``n_frames`` non-zero 1280-sample frames and collect the per-frame fire bools."""
    frame = np.full(1280, 0.1, dtype=np.float32)
    return [det.detect(frame) for _ in range(n_frames)]


# --------------------------------------------------------------------------- #
# (1) window == 1 is byte-identical to the old single-frame decision          #
# --------------------------------------------------------------------------- #
def test_window_one_is_single_frame_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    _make_scripted(monkeypatch, [0.1, 0.5, 0.3, 0.9, 0.0])
    det = WakeWord("wakewords/w.onnx", threshold=0.4, phases=1, window=1, refractory=0)
    fires = _drive(det, 5)
    # Fire exactly when the single frame's score >= 0.4 — the classic behaviour.
    assert fires == [False, True, False, True, False]


def test_window_one_default_unspecified(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare WakeWord(...) defaults to window=1 / refractory=0 (no behaviour change)."""
    _make_scripted(monkeypatch, [0.9])
    det = WakeWord("wakewords/w.onnx", threshold=0.4)
    assert det.window == 1 and det.refractory == 0
    assert det.detect(np.full(1280, 0.1, dtype=np.float32)) is True


# --------------------------------------------------------------------------- #
# (2) moving-average fire: mean >= threshold, dip tolerated, spike rejected   #
# --------------------------------------------------------------------------- #
def test_moving_average_fires_on_mean(monkeypatch: pytest.MonkeyPatch) -> None:
    """window=2 fires when the MEAN of the last two frames clears the threshold."""
    # Means at thr 0.5: [0.3], [0.35], [0.5*]=fires. (0.4+0.6)/2 = 0.5.
    _make_scripted(monkeypatch, [0.3, 0.4, 0.6])
    det = WakeWord("wakewords/w.onnx", threshold=0.5, phases=1, window=2)
    assert _drive(det, 3) == [False, False, True]


def test_window_tolerates_single_frame_dip(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-frame dip below threshold still fires under a window that averages it out —
    microWakeWord's smoothing (a single sub-threshold frame doesn't kill a strong word)."""
    # Single-frame: 0.7, 0.2(dip), 0.7. window=2 means: -, 0.45, 0.45 — would NOT fire.
    # But with a strong word the dip is shallow: 0.7, 0.5, 0.7 -> means 0.6, 0.6 fire at 0.55.
    _make_scripted(monkeypatch, [0.7, 0.5, 0.7])
    det = WakeWord("wakewords/w.onnx", threshold=0.55, phases=1, window=2)
    fires = _drive(det, 3)
    assert fires[1] is True  # the dip frame's window mean (0.6) still clears 0.55


def test_window_rejects_lone_spike(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single high spike surrounded by low frames does NOT fire under window>1 (the mean
    stays below threshold) — the false-positive-suppression half of the smoothing."""
    # 0.0, 0.9(spike), 0.0. window=2 means: -, 0.45, 0.45 — never reaches 0.5.
    _make_scripted(monkeypatch, [0.0, 0.9, 0.0])
    det = WakeWord("wakewords/w.onnx", threshold=0.5, phases=1, window=2)
    assert _drive(det, 3) == [False, False, False]
    # The same lone spike DOES fire at window=1 (single-frame) — proving the window matters.
    _make_scripted(monkeypatch, [0.0, 0.9, 0.0])
    det1 = WakeWord("wakewords/w.onnx", threshold=0.5, phases=1, window=1)
    assert _drive(det1, 3) == [False, True, False]


# --------------------------------------------------------------------------- #
# (3) refractory lockout suppresses re-fires for N frames after a fire        #
# --------------------------------------------------------------------------- #
def test_refractory_suppresses_refires(monkeypatch: pytest.MonkeyPatch) -> None:
    """After a fire, the next ``refractory`` frames cannot fire even if above threshold."""
    # 5 consecutive high frames, window=1, refractory=2: fire @0, lock @1,2, fire @3, lock @4.
    _make_scripted(monkeypatch, [0.9, 0.9, 0.9, 0.9, 0.9])
    det = WakeWord("wakewords/w.onnx", threshold=0.4, phases=1, window=1, refractory=2)
    assert _drive(det, 5) == [True, False, False, True, False]


def test_refractory_zero_fires_every_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """refractory=0 (default) puts no lockout between consecutive above-threshold fires."""
    _make_scripted(monkeypatch, [0.9, 0.9, 0.9])
    det = WakeWord("wakewords/w.onnx", threshold=0.4, phases=1, window=1, refractory=0)
    assert _drive(det, 3) == [True, True, True]


# --------------------------------------------------------------------------- #
# (4) reset() clears BOTH the moving-average window and the refractory        #
# --------------------------------------------------------------------------- #
def test_reset_clears_window_and_refractory(monkeypatch: pytest.MonkeyPatch) -> None:
    _make_scripted(monkeypatch, [0.9, 0.1, 0.1])
    det = WakeWord("wakewords/w.onnx", threshold=0.4, phases=1, window=3, refractory=5)
    det.detect(np.full(1280, 0.1, dtype=np.float32))  # fire -> window=[0.9], refractory armed
    assert det._refractory_left == 5
    assert len(det._score_window) == 1
    det.reset()
    assert det._refractory_left == 0
    assert len(det._score_window) == 0
    assert det.last_score == 0.0
    # After reset, a fresh high frame fires immediately (the old high score is gone, and the
    # refractory no longer locks it out).
    _make_scripted(monkeypatch, [0.9])
    det2 = WakeWord("wakewords/w.onnx", threshold=0.4, phases=1, window=3, refractory=5)
    det2._refractory_left = 3  # pretend mid-lockout
    det2._score_window.append(0.0)
    det2.reset()
    assert det2.detect(np.full(1280, 0.1, dtype=np.float32)) is True


# --------------------------------------------------------------------------- #
# (5) pure eval-path helpers: count_fires_moving_average / moving_average_fires #
# --------------------------------------------------------------------------- #
def test_count_fires_moving_average_window_one_refractory_zero() -> None:
    """window=1 + refractory=0 == one fire per above-threshold frame (single-frame)."""
    trace = [0.1, 0.9, 0.9, 0.1, 0.9]
    assert count_fires_moving_average(trace, 0.4, window=1, refractory=0) == 3


def test_count_fires_moving_average_refractory_groups() -> None:
    """A sustained above-threshold passage collapses to refractory-spaced fires."""
    trace = [0.9] * 10
    # window=1, refractory=4: fire @0, lock 1-4, fire @5, lock 6-9 -> 2 fires.
    assert count_fires_moving_average(trace, 0.4, window=1, refractory=4) == 2


def test_count_fires_moving_average_window_smooths() -> None:
    """A lone spike is NOT a fire under window=2 (mean stays low); a sustained run is."""
    assert count_fires_moving_average([0.0, 0.9, 0.0], 0.5, window=2, refractory=0) == 0
    assert count_fires_moving_average([0.0, 0.6, 0.6], 0.5, window=2, refractory=0) >= 1


def test_moving_average_fires_is_count_gt_zero() -> None:
    assert moving_average_fires([0.0, 0.9, 0.0], 0.5, window=2) is False
    assert moving_average_fires([0.0, 0.6, 0.6], 0.5, window=2) is True
    # window=1 reproduces "any frame >= threshold".
    assert moving_average_fires([0.0, 0.9, 0.0], 0.5, window=1) is True


def test_empty_trace_no_fire() -> None:
    assert count_fires_moving_average([], 0.4, window=2, refractory=3) == 0
    assert moving_average_fires([], 0.4, window=2) is False


# --------------------------------------------------------------------------- #
# (6) eval-path consistency: score_wake_clip uses the SAME criterion          #
# --------------------------------------------------------------------------- #
def test_score_wake_clip_uses_moving_average(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:  # noqa: ANN001
    """score_wake_clip with window/refractory must reproduce moving_average_fires on the
    SAME trace — proving the offline eval matches the live detector (live == eval)."""
    from my_stt_tts import wake as wake_mod

    # A lone spike trace: single-frame fires, window=2 does not. Build a 5-frame clip and a
    # model that scores high ONLY on the middle frame, so the per-frame trace is a lone spike.
    class SpikeModel:
        _n = 0

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def predict(self, _frame: np.ndarray) -> dict[str, float]:
            cls = type(self)
            cls._n += 1
            return {"w": 0.9 if cls._n == 3 else 0.0}

        def reset(self) -> None:
            return None

    model_path = tmp_path / "w.onnx"
    model_path.write_bytes(b"stub")  # only existence is checked; the model is mocked
    monkeypatch.setattr(wake_mod, "wake_model_for", lambda *_a, **_k: str(model_path))

    def fresh_spike(**_k: Any) -> type:
        return type("S", (SpikeModel,), {"_n": 0})

    clip = np.tile(np.full(1280, 0.1, dtype=np.float32), 6)  # 6 frames at phases=1

    # phases=1 so the trace is one score per frame (matching the scripted model).
    _install_fake_openwakeword(monkeypatch, fresh_spike())
    conf1, fired1, trace1 = wake_mod.score_wake_clip(
        clip, 16000, "w", threshold=0.5, phases=1, window=1, with_trace=True
    )
    _install_fake_openwakeword(monkeypatch, fresh_spike())
    _conf2, fired2, trace2 = wake_mod.score_wake_clip(
        clip, 16000, "w", threshold=0.5, phases=1, window=2, with_trace=True
    )
    assert conf1 == pytest.approx(0.9)
    # window=1 fires on the lone spike; window=2 does not (the mean is diluted) — and each
    # equals what moving_average_fires says on the very same trace.
    assert fired1 is True
    assert fired1 == moving_average_fires(trace1, 0.5, window=1)
    assert fired2 is False
    assert fired2 == moving_average_fires(trace2, 0.5, window=2)


# --------------------------------------------------------------------------- #
# (7) fa_eval honours window/refractory (live == eval on the FA side too)     #
# --------------------------------------------------------------------------- #
def test_fa_eval_window_refractory_changes_counts() -> None:
    from my_stt_tts.wake import fa_eval

    # One negative trace: a 10-frame sustained 0.9 passage. At thr 0.5:
    #  default (window=1, refractory=0) -> count_fa_events groups it to 1 event.
    #  window=1, refractory=4 -> count_fires_moving_average -> 2 fires (refractory-spaced).
    neg = [[0.9] * 10]
    base = fa_eval([], neg, thresholds=[0.5])
    refr = fa_eval([], neg, thresholds=[0.5], refractory=4)
    assert base["points"][0]["fa_per_hour"] > 0
    # The refractory path counts the live re-fires; the default path counts merged events.
    assert refr["points"][0]["fa_per_hour"] != base["points"][0]["fa_per_hour"]


def test_fa_eval_recall_uses_moving_average_when_smoothing() -> None:
    from my_stt_tts.wake import fa_eval

    # A positive whose trace is a lone spike: recall=1 at window=1 (any frame >= thr) but
    # recall=0 at window=2 (mean diluted) — the recall side must track the live criterion.
    pos = [[0.0, 0.9, 0.0]]
    r1 = fa_eval(pos, [[0.0]], thresholds=[0.5], window=1)
    r2 = fa_eval(pos, [[0.0]], thresholds=[0.5], window=2)
    assert r1["points"][0]["true_accept"] == 1.0
    assert r2["points"][0]["true_accept"] == 0.0


# --------------------------------------------------------------------------- #
# (8) config knobs: default / env / validate / settings round-trip           #
# --------------------------------------------------------------------------- #
def test_config_defaults_window_one_refractory_eight() -> None:
    from my_stt_tts.config import Config

    cfg = Config(anthropic_api_key="x")
    assert cfg.wake_window == 1  # byte-identical default (the no-regression gate)
    assert cfg.wake_refractory == 8


def test_config_from_env_parses_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    from my_stt_tts.config import Config

    monkeypatch.setenv("WAKE_WINDOW", "3")
    monkeypatch.setenv("WAKE_REFRACTORY", "5")
    cfg = Config.from_env()
    assert cfg.wake_window == 3 and cfg.wake_refractory == 5


def test_config_validate_bounds() -> None:
    from my_stt_tts.config import Config, ConfigError

    # In-bounds config validates cleanly (no raise for the wake knobs).
    Config(anthropic_api_key="x", wake_window=1, wake_refractory=0).validate()
    bad = Config(anthropic_api_key="x", wake_window=0, wake_refractory=-1)
    with pytest.raises(ConfigError) as exc:
        bad.validate()
    msg = str(exc.value)
    assert "wake_window" in msg and "wake_refractory" in msg


def test_settings_dict_and_apply_round_trip() -> None:
    from my_stt_tts.config import Config
    from my_stt_tts.webui import apply_settings, settings_dict

    cfg = Config(anthropic_api_key="x")
    d = settings_dict(cfg)
    assert d["wake_window"] == 1 and d["wake_refractory"] == 8
    apply_settings(cfg, {"wake_window": 4, "wake_refractory": 12})
    assert cfg.wake_window == 4 and cfg.wake_refractory == 12
    # Clamps: window floored at 1 / capped at 50; refractory floored at 0.
    apply_settings(cfg, {"wake_window": 0, "wake_refractory": -3})
    assert cfg.wake_window == 1 and cfg.wake_refractory == 0


def test_settings_text_shows_window_refractory() -> None:
    from my_stt_tts.__main__ import settings_text
    from my_stt_tts.config import Config

    text = settings_text(Config(anthropic_api_key="x"), color=False)
    assert "window 1" in text and "refractory 8" in text


def test_from_config_threads_knobs(monkeypatch: pytest.MonkeyPatch) -> None:
    """WakeWord.from_config wires cfg.wake_window / cfg.wake_refractory into the detector."""
    _make_scripted(monkeypatch, [0.0])
    from my_stt_tts.config import Config

    cfg = Config(anthropic_api_key="x", wake_window=3, wake_refractory=7)
    det = WakeWord.from_config(cfg)
    assert det.window == 3 and det.refractory == 7
