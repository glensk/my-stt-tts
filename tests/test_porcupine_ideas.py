"""Porcupine-ideas (repo #4 of the wake-detection checker loop) — Python side.

Picovoice Porcupine's engine is closed; we port the two PORTABLE IDEAS onto our open
machinery:

* Feature 1 — a unified 0..1 ``wake_sensitivity`` knob that maps onto the openWakeWord
  ``wake_threshold`` via the MEASURED fa_eval curve (curve inversion) or a documented
  linear fallback; per-word overrides; the ``guidance`` hint.
* Feature 2 — a noise×SNR benchmark harness: ``mix_at_snr`` (RMS-energy-matched),
  the ``per_snr`` axis on ``fa_eval`` / the event, adaptive threshold bracketing that
  brackets ``target_fa`` where a fixed linspace grid would clamp.

All math is exercised with synthetic clips / mocked openWakeWord (no GPU, no wheel).
"""
# pylint: disable=missing-function-docstring,protected-access,too-few-public-methods
# pylint: disable=missing-class-docstring,import-outside-toplevel

from __future__ import annotations

import json
import queue
import sys
import types
from typing import Any

import numpy as np
import pytest

from my_stt_tts.audio import mix_at_snr
from my_stt_tts.config import (
    SENSITIVITY_THRESHOLD_MAX,
    SENSITIVITY_THRESHOLD_MIN,
    Config,
    ConfigError,
    _parse_sensitivity_map,
    sensitivity_to_threshold,
    wake_word_guidance,
)
from my_stt_tts.events import EventBus
from my_stt_tts.wake import (
    DEFAULT_SNR_LIST,
    SECONDS_PER_FRAME,
    _adaptive_threshold_grid,
    fa_eval,
    fa_eval_snr,
)


# --------------------------------------------------------------------------- #
# Feature 1 — sensitivity_to_threshold: linear fallback + curve inversion      #
# --------------------------------------------------------------------------- #
def test_sensitivity_linear_fallback_endpoints_and_midpoint() -> None:
    # No curve -> uncalibrated linear remap (inverted: higher sensitivity = lower thr).
    thr0, cal0 = sensitivity_to_threshold("w", 0.0)
    thr_half, cal_half = sensitivity_to_threshold("w", 0.5)
    thr1, cal1 = sensitivity_to_threshold("w", 1.0)
    assert (cal0, cal_half, cal1) == (False, False, False)
    assert thr0 == pytest.approx(SENSITIVITY_THRESHOLD_MAX)  # strictest
    assert thr1 == pytest.approx(SENSITIVITY_THRESHOLD_MIN)  # loosest
    assert thr_half == pytest.approx((SENSITIVITY_THRESHOLD_MIN + SENSITIVITY_THRESHOLD_MAX) / 2.0)


def test_sensitivity_linear_monotonic_decreasing() -> None:
    thrs = [sensitivity_to_threshold("w", s)[0] for s in np.linspace(0, 1, 11)]
    # Higher sensitivity never RAISES the threshold (fires at least as easily).
    assert all(b <= a + 1e-9 for a, b in zip(thrs, thrs[1:], strict=False))


def test_sensitivity_clamps_out_of_range() -> None:
    # Out-of-range sensitivity clamps to [0,1] before mapping; threshold stays in [0,1].
    assert sensitivity_to_threshold("w", -2.0)[0] == pytest.approx(SENSITIVITY_THRESHOLD_MAX)
    assert sensitivity_to_threshold("w", 5.0)[0] == pytest.approx(SENSITIVITY_THRESHOLD_MIN)
    for s in (-1.0, 0.0, 0.3, 1.0, 9.0):
        thr, _ = sensitivity_to_threshold("w", s)
        assert 0.0 <= thr <= 1.0


def _measured_curve() -> dict[str, Any]:
    # STRICT (low FA, high thr) -> LOOSE (high FA, low thr).
    return {
        "points": [
            {"threshold": 0.9, "fa_per_hour": 0.0, "true_accept": 0.2},
            {"threshold": 0.6, "fa_per_hour": 0.5, "true_accept": 0.6},
            {"threshold": 0.4, "fa_per_hour": 2.0, "true_accept": 0.85},
            {"threshold": 0.1, "fa_per_hour": 12.0, "true_accept": 1.0},
        ]
    }


def test_sensitivity_curve_inversion_walks_operating_points() -> None:
    curve = _measured_curve()
    thr0, cal0 = sensitivity_to_threshold("maziko", 0.0, curve=curve)
    thr_mid, cal_mid = sensitivity_to_threshold("maziko", 0.5, curve=curve)
    thr1, cal1 = sensitivity_to_threshold("maziko", 1.0, curve=curve)
    assert (cal0, cal_mid, cal1) == (True, True, True)
    assert thr0 == pytest.approx(0.9)  # strictest measured operating point
    assert thr1 == pytest.approx(0.1)  # loosest measured operating point
    # 0.5 sits on the interior (between the two middle points), not at an endpoint.
    assert 0.1 < thr_mid < 0.9


def test_sensitivity_curve_inversion_monotonic() -> None:
    curve = _measured_curve()
    thrs = [sensitivity_to_threshold("w", s, curve=curve)[0] for s in np.linspace(0, 1, 21)]
    assert all(b <= a + 1e-9 for a, b in zip(thrs, thrs[1:], strict=False))


def test_sensitivity_falls_back_when_curve_too_thin() -> None:
    # A 1-point curve cannot be inverted -> linear fallback, calibrated False.
    thin = {"points": [{"threshold": 0.5, "fa_per_hour": 1.0, "true_accept": 0.5}]}
    thr, cal = sensitivity_to_threshold("w", 0.3, curve=thin)
    assert cal is False
    assert thr == sensitivity_to_threshold("w", 0.3)[0]
    # Empty / None curve too.
    assert sensitivity_to_threshold("w", 0.3, curve={"points": []})[1] is False
    assert sensitivity_to_threshold("w", 0.3, curve=None)[1] is False


def test_sensitivity_accepts_bare_points_list() -> None:
    thr, cal = sensitivity_to_threshold("w", 0.0, curve=_measured_curve()["points"])
    assert cal is True and thr == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# Feature 1 — per-word sensitivity map parse + validate                        #
# --------------------------------------------------------------------------- #
def test_parse_sensitivity_map_basic() -> None:
    out = _parse_sensitivity_map("hey_jarvis=0.4;maziko=0.8")
    assert out == {"hey_jarvis": 0.4, "maziko": 0.8}


def test_parse_sensitivity_map_clamps_and_skips_garbage() -> None:
    out = _parse_sensitivity_map("a=2.0; b=-1; c=notnum; =0.5; nope; d=0.3")
    # 2.0 -> 1.0, -1 -> 0.0; the non-numeric, empty-key, and no-'=' groups are skipped.
    assert out == {"a": 1.0, "b": 0.0, "d": 0.3}


def test_parse_sensitivity_map_empty() -> None:
    assert _parse_sensitivity_map("") == {}
    assert _parse_sensitivity_map(None) == {}
    assert _parse_sensitivity_map("   ") == {}


def test_config_set_wake_sensitivity_global(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = Config()
    cfg.set_wake_sensitivity_env("0.2")
    assert cfg.wake_sensitivity == pytest.approx(0.2)
    assert cfg.wake_sensitivity_set is True
    assert cfg.wake_sensitivity_map == {}


def test_config_set_wake_sensitivity_map() -> None:
    cfg = Config()
    cfg.set_wake_sensitivity_env("hey_jarvis=0.4;maziko=0.8")
    assert cfg.wake_sensitivity_set is True
    assert cfg.wake_sensitivity_map == {"hey_jarvis": 0.4, "maziko": 0.8}
    # Per-word override beats the global; absent word falls back to global.
    assert cfg.sensitivity_for("maziko") == pytest.approx(0.8)
    assert cfg.sensitivity_for("unknown") == pytest.approx(cfg.wake_sensitivity)


def test_config_sensitivity_derives_threshold_master() -> None:
    cfg = Config()
    cfg.wake_phrase = "hey_jarvis"
    cfg.set_wake_sensitivity_env("0.0")  # strictest
    thr, cal = cfg.derive_wake_threshold()
    assert cal is False  # no curve here
    assert cfg.wake_threshold == pytest.approx(SENSITIVITY_THRESHOLD_MAX)
    assert thr == pytest.approx(SENSITIVITY_THRESHOLD_MAX)


def test_config_threshold_master_when_sensitivity_unset() -> None:
    cfg = Config()
    cfg.wake_threshold = 0.4
    # Sensitivity NOT set -> derive_wake_threshold is a no-op; explicit threshold stands.
    thr, cal = cfg.derive_wake_threshold()
    assert (thr, cal) == (0.4, False)
    assert cfg.wake_threshold == 0.4


def test_config_validate_rejects_out_of_range_sensitivity() -> None:
    cfg = Config(anthropic_api_key="x")
    cfg.wake_sensitivity = 1.5
    with pytest.raises(ConfigError, match="wake_sensitivity"):
        cfg.validate()
    cfg.wake_sensitivity = 0.5
    cfg.wake_sensitivity_map = {"maziko": 3.0}
    with pytest.raises(ConfigError, match="maziko"):
        cfg.validate()


def test_from_env_wake_sensitivity_overrides_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    # WAKE_SENSITIVITY set ALONGSIDE WAKE_THRESHOLD: sensitivity wins (documented master).
    monkeypatch.setenv("WAKE_PHRASE", "hey_jarvis")
    monkeypatch.setenv("WAKE_THRESHOLD", "0.4")
    monkeypatch.setenv("WAKE_SENSITIVITY", "1.0")  # loosest -> MIN threshold
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = Config.from_env()
    assert cfg.wake_sensitivity_set is True
    assert cfg.wake_threshold == pytest.approx(SENSITIVITY_THRESHOLD_MIN)


def test_from_env_noise_corpus_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAKE_NOISE_CORPUS", "/tmp/my-noise")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    cfg = Config.from_env()
    assert cfg.noise_corpus_dir == "/tmp/my-noise"


# --------------------------------------------------------------------------- #
# Feature 1 — guidance strings                                                 #
# --------------------------------------------------------------------------- #
def test_guidance_red_tier_raise() -> None:
    assert (
        wake_word_guidance("nexus", tier="red", measured=False, stats={})
        == "Missing it? Raise sensitivity."
    )


def test_guidance_recent_false_accept_lower() -> None:
    # A recent test that FIRED on near-silence (conf <= 0.05) -> too sensitive.
    stats = {"maziko": [{"fired": True, "confidence": 0.01, "source": "server"}]}
    assert (
        wake_word_guidance("maziko", tier="green", measured=True, stats=stats)
        == "Firing on its own? Lower sensitivity."
    )


def test_guidance_fa_takes_priority_over_red() -> None:
    # A spurious fire signal wins even if the tier is red.
    stats = {"sage": [{"fired": True, "confidence": 0.0, "source": "server"}]}
    assert (
        wake_word_guidance("sage", tier="red", measured=False, stats=stats)
        == "Firing on its own? Lower sensitivity."
    )


def test_guidance_healthy_word_empty() -> None:
    assert wake_word_guidance("hey_jarvis", tier="green", measured=False, stats={}) == ""
    # A high-confidence real fire is NOT a false accept.
    stats = {"hey_jarvis": [{"fired": True, "confidence": 0.9, "source": "server"}]}
    assert wake_word_guidance("hey_jarvis", tier="green", measured=True, stats=stats) == ""


# --------------------------------------------------------------------------- #
# Feature 2 — mix_at_snr RMS-energy match                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("target", [20.0, 10.0, 5.0, 0.0, -5.0])
def test_mix_at_snr_matches_target(target: float) -> None:
    rng = np.random.default_rng(7)
    speech = (rng.standard_normal(16000).astype(np.float32)) * 0.1
    noise = (rng.standard_normal(4000).astype(np.float32)) * 0.5
    # Reconstruct the scale the mixer uses and confirm the achieved SNR == target.
    reps = int(np.ceil(speech.size / noise.size))
    nzt = np.tile(noise, reps)[: speech.size]
    e_s = float(np.sum(speech.astype(np.float64) ** 2))
    e_n = float(np.sum(nzt.astype(np.float64) ** 2))
    scale = np.sqrt(e_n * 10 ** (target / 10) / e_s)
    achieved = 10 * np.log10((e_s * scale**2) / e_n)
    assert achieved == pytest.approx(target, abs=1e-6)
    # The mixer output is the speech length, float32, clip-protected.
    mixed = mix_at_snr(speech, noise, target)
    assert mixed.dtype == np.float32
    assert mixed.size == speech.size
    assert float(np.max(np.abs(mixed))) <= 1.0


def test_mix_at_snr_tiles_short_noise() -> None:
    speech = np.ones(1000, dtype=np.float32) * 0.1
    noise = np.array([0.2, -0.2], dtype=np.float32)
    mixed = mix_at_snr(speech, noise, 10.0)
    assert mixed.size == 1000  # noise tiled to length


def test_mix_at_snr_trims_long_noise() -> None:
    speech = np.ones(500, dtype=np.float32) * 0.1
    noise = np.ones(5000, dtype=np.float32) * 0.2
    assert mix_at_snr(speech, noise, 10.0).size == 500


def test_mix_at_snr_degenerate() -> None:
    sp = np.ones(100, dtype=np.float32) * 0.1
    nz = np.ones(50, dtype=np.float32) * 0.2
    assert mix_at_snr(np.array([], dtype=np.float32), nz, 10.0).size == 0
    # Silent noise -> speech unchanged.
    assert np.allclose(mix_at_snr(sp, np.zeros(20, dtype=np.float32), 10.0), sp)
    # Silent speech -> length-matched noise alone.
    out = mix_at_snr(np.zeros(50, dtype=np.float32), nz, 10.0)
    assert np.allclose(out, np.clip(nz, -1, 1))


# --------------------------------------------------------------------------- #
# Feature 2 — adaptive threshold bracketing                                    #
# --------------------------------------------------------------------------- #
def test_adaptive_bracketing_reaches_target_where_linspace_clamps() -> None:
    # A graded ROC whose crossings all sit ABOVE 0.95: every linspace point (max 0.95)
    # still leaves false-accepts, so the linspace minimum FA is far above target 0.5 and
    # np.interp CLAMPS. The adaptive grid reaches thr ~1.0 where FA truly hits 0.
    neg = [[lvl] for lvl in np.linspace(0.905, 0.999, 60)]
    pos = [[lvl] for lvl in np.linspace(0.90, 0.999, 40)]
    fixed = fa_eval(pos, neg, target_fa=0.5, adaptive=False)
    adapt = fa_eval(pos, neg, target_fa=0.5, adaptive=True)
    fa_fixed_min = min(p["fa_per_hour"] for p in fixed["points"])
    fa_adapt_min = min(p["fa_per_hour"] for p in adapt["points"])
    assert fa_fixed_min > 0.5  # linspace never reaches the budget -> would clamp
    assert fa_adapt_min == pytest.approx(0.0)  # adaptive reaches it
    # The honest answer at the 0.5 FA budget is total miss; the clamped fixed grid lies.
    assert adapt["miss_at_target_fa"] == pytest.approx(1.0)
    assert fixed["miss_at_target_fa"] < adapt["miss_at_target_fa"]
    assert len(adapt["points"]) > len(fixed["points"])


def test_adaptive_grid_keeps_event_grouped_counting() -> None:
    # A sustained above-threshold passage is ONE event, not one per frame (the honest
    # count we must not regress). 20 contiguous frames at 0.6 over 1.6 s -> 1 event.
    neg = [[0.6] * 20]
    grid = _adaptive_threshold_grid(
        neg,
        20 * SECONDS_PER_FRAME,
        0.5,
        grouping_window=10,
        window=1,
        refractory=0,
        smoothing=False,
        base_grid=[round(t, 3) for t in np.linspace(0.05, 0.95, 19)],
    )
    # Endpoints 0.0 and 1.0 are always added so the budget is bounded on both sides.
    assert 0.0 in grid and 1.0 in grid
    # FA at thr 0.5 counts 1 event (not 20) -> 1/1.6s*3600 = 2250/hr.
    r = fa_eval(neg_traces=neg, pos_traces=[], target_fa=0.5)
    fa_at_low = [p["fa_per_hour"] for p in r["points"] if p["threshold"] <= 0.5][0]
    assert fa_at_low == pytest.approx(20 / (20 * SECONDS_PER_FRAME) / 20 * 3600.0)


def test_adaptive_empty_negatives_returns_base_grid() -> None:
    base = [round(t, 3) for t in np.linspace(0.05, 0.95, 19)]
    out = _adaptive_threshold_grid(
        [], 0.0, 0.5, grouping_window=10, window=1, refractory=0, smoothing=False, base_grid=base
    )
    assert out == sorted(set(base))


# --------------------------------------------------------------------------- #
# Feature 2 — fa_eval_snr: per_snr shape + empty-noise graceful                #
# --------------------------------------------------------------------------- #
class _EnergyModel:
    """Scores high on loud int16 PCM, ~0 on silence — deterministic stand-in for oWW."""

    def __init__(self, **_kwargs: Any) -> None:
        pass

    def predict(self, frame: np.ndarray) -> dict[str, float]:
        return {"maziko": 0.9 if int(np.max(np.abs(frame))) > 1000 else 0.001}


def _install_fake_oww(monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = types.ModuleType("openwakeword")
    mod = types.ModuleType("openwakeword.model")
    mod.Model = _EnergyModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openwakeword", pkg)
    monkeypatch.setitem(sys.modules, "openwakeword.model", mod)


def test_fa_eval_snr_default_list_constant() -> None:
    assert DEFAULT_SNR_LIST == (None, 10.0, 5.0)


def test_fa_eval_snr_empty_noise_clean_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _install_fake_oww(monkeypatch)
    # A present (fake) model file so score_wake_clip runs.
    (tmp_path / "maziko.onnx").write_bytes(b"x")
    pos = [np.ones(16000, dtype=np.float32) * 0.5]
    neg = [np.ones(16000, dtype=np.float32) * 0.5]
    out = fa_eval_snr(
        pos, neg, [], "maziko", wakewords_dir=str(tmp_path), snr_list=[None, 10.0, 5.0]
    )
    # Empty noise collapses to clean-only regardless of the requested SNR list.
    assert out["snr_list"] == [None]
    assert len(out["per_snr"]) == 1
    assert out["per_snr"][0]["snr_db"] is None
    assert "points" in out["per_snr"][0]
    assert out["clean"]  # the clean condition is populated


def test_fa_eval_snr_with_noise_per_snr_shape(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    _install_fake_oww(monkeypatch)
    (tmp_path / "maziko.onnx").write_bytes(b"x")
    pos = [np.ones(16000, dtype=np.float32) * 0.5]
    neg = [np.ones(16000, dtype=np.float32) * 0.5]
    noise = [np.random.default_rng(0).standard_normal(8000).astype(np.float32) * 0.3]
    out = fa_eval_snr(
        pos, neg, noise, "maziko", wakewords_dir=str(tmp_path), snr_list=[None, 10.0, 5.0]
    )
    assert out["snr_list"] == [None, 10.0, 5.0]
    assert [e["snr_db"] for e in out["per_snr"]] == [None, 10.0, 5.0]
    for entry in out["per_snr"]:
        assert set(entry) == {"snr_db", "miss_at_target_fa", "points"}
        assert isinstance(entry["points"], list) and entry["points"]
        for p in entry["points"]:
            assert set(p) == {"threshold", "fa_per_hour", "true_accept"}


# --------------------------------------------------------------------------- #
# Feature 2 — fa_eval_result event carries optional per_snr + snr_list          #
# --------------------------------------------------------------------------- #
def _drain(sub: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while True:
        try:
            out.append(json.loads(sub.get_nowait()))
        except queue.Empty:
            break
    return out


def test_fa_eval_event_with_per_snr() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _drain(sub)  # clear the replayed state frame
    bus.fa_eval_result(
        word="maziko",
        points=[{"threshold": 0.5, "fa_per_hour": 1.0, "true_accept": 0.8}],
        miss_at_target_fa=0.2,
        target_fa=0.5,
        neg_seconds=16.0,
        per_snr=[
            {
                "snr_db": None,
                "miss_at_target_fa": 0.2,
                "points": [{"threshold": 0.5, "fa_per_hour": 1.0, "true_accept": 0.8}],
            },
            {
                "snr_db": 5.0,
                "miss_at_target_fa": 0.5,
                "points": [{"threshold": 0.5, "fa_per_hour": 3.0, "true_accept": 0.5}],
            },
        ],
        snr_list=[None, 5.0],
        message="ok",
    )
    events = [e for e in _drain(sub) if e.get("type") == "fa_eval_result"]
    assert len(events) == 1
    e = events[0]
    assert e["snr_list"] == [None, 5.0]
    assert [p["snr_db"] for p in e["per_snr"]] == [None, 5.0]
    assert e["per_snr"][1]["miss_at_target_fa"] == pytest.approx(0.5)


def test_fa_eval_event_without_per_snr_back_compat() -> None:
    bus = EventBus()
    sub = bus.subscribe()
    _drain(sub)
    bus.fa_eval_result(
        word="hey_jarvis",
        points=[{"threshold": 0.5, "fa_per_hour": 0.0, "true_accept": 1.0}],
        miss_at_target_fa=0.0,
        target_fa=0.5,
    )
    e = [x for x in _drain(sub) if x.get("type") == "fa_eval_result"][0]
    # No noise corpus -> both fields present but null (base contract unchanged).
    assert e["per_snr"] is None
    assert e["snr_list"] is None
