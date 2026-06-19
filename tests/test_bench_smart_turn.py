"""Tests for G4 — the Smart-Turn latency bench logic.

The bench's pure logic (percentiles, the budget = headroom × silence-window, the
p95-fits-the-window assertion, and the timed run loop driven by an injectable
clock) is tested here with a FAKE clock and a FAKE inference callable — no ONNX,
no model, no wall-clock sleeps. The on-device timing path is exercised by running
the real script (which reports ``skipped`` without the model).
"""
# pylint: disable=missing-function-docstring,import-outside-toplevel

import importlib.util
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1] / "scripts" / "bench_smart_turn.py"
_spec = importlib.util.spec_from_file_location("bench_smart_turn", _BENCH)
assert _spec is not None and _spec.loader is not None
bench = importlib.util.module_from_spec(_spec)
sys.modules["bench_smart_turn"] = bench
_spec.loader.exec_module(bench)


def test_percentile():
    assert bench.percentile([], 50) == 0.0
    assert bench.percentile([5.0], 95) == 5.0
    assert bench.percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert bench.percentile([1.0, 2.0, 3.0, 4.0], 95) == 3.85


def test_budget_is_headroom_times_window():
    res = bench.BenchResult(silence_window_ms=700.0, headroom=0.5)
    assert res.budget_ms == 350.0
    res2 = bench.BenchResult(silence_window_ms=700.0, headroom=0.25)
    assert res2.budget_ms == 175.0


def test_fits_silence_window_pass():
    res = bench.BenchResult(latencies_ms=[40.0] * 10, silence_window_ms=700.0, headroom=0.5)
    assert bench.fits_silence_window(res)  # p95 40 ms << 350 ms budget


def test_fits_silence_window_fail_when_too_slow():
    res = bench.BenchResult(latencies_ms=[400.0] * 10, silence_window_ms=700.0, headroom=0.5)
    assert not bench.fits_silence_window(res)  # p95 400 ms > 350 ms budget


def test_fits_silence_window_uses_p95_not_mean():
    # Mostly fast, but a slow tail the p95 must catch (400 ms > 350 ms budget).
    lat = [30.0] * 90 + [400.0] * 10
    res = bench.BenchResult(latencies_ms=lat, silence_window_ms=700.0, headroom=0.5)
    assert res.p95 >= 350.0
    assert not bench.fits_silence_window(res)


def test_empty_result_is_failure():
    assert not bench.fits_silence_window(bench.BenchResult(latencies_ms=[]))


def test_run_bench_with_fake_clock():
    # A deterministic clock advancing 0.05 s per tick -> each inference "takes" 50 ms.
    ticks = iter(range(10_000))
    clock = lambda: next(ticks) * 0.05  # noqa: E731

    def _infer():
        return 0.9

    res = bench.run_bench(_infer, runs=5, silence_window_ms=700.0, headroom=0.5, clock=clock)
    assert len(res.latencies_ms) == 5
    assert all(abs(x - 50.0) < 1e-6 for x in res.latencies_ms)
    assert bench.fits_silence_window(res)


def test_summarize_shape():
    res = bench.BenchResult(latencies_ms=[20.0, 30.0, 25.0], silence_window_ms=700.0, headroom=0.5)
    s = bench.summarize(res)
    assert set(s) == {
        "runs",
        "warm_ms",
        "p50_ms",
        "p95_ms",
        "silence_window_ms",
        "budget_ms",
        "fits_silence_window",
    }
    assert s["runs"] == 3
    assert s["fits_silence_window"] is True


def test_main_skips_gracefully_without_model(capsys):
    # No onnxruntime/model on the test host -> the bench reports skipped + exits 0.
    rc = bench.main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "skipped" in out
