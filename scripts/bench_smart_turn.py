#!/usr/bin/env python3
"""Benchmark Smart Turn v3 end-of-turn inference latency on THIS machine (G4).

The smart endpointer (``turn.SmartTurnAnalyzer``) runs the Smart Turn v3 ONNX model
on a *silence candidate* to decide whether a pause is really end-of-turn. For that
to feel natural the model's inference must comfortably fit inside the silence
window the loop waits before asking it (``vad_silence_seconds``, default 0.7 s):
if inference took longer than the window, the assistant would either cut the user
off or stall. This bench measures the real on-device inference latency (warm + a
percentile sweep) and **asserts** it fits the window with headroom.

The assertion logic (:func:`fits_silence_window`, :func:`summarize`) is pure so it
is unit-tested with a fake clock + fake session (no model, no ONNX) — see
``tests/test_bench_smart_turn.py``. When the model / ``onnxruntime`` are genuinely
installed it times the real thing; otherwise it reports ``skipped`` and exits 0 so
the bench is safe to run anywhere.

Usage:
    uv run scripts/bench_smart_turn.py [--runs 30] [--headroom 0.5] [--json]
"""
# pylint: disable=broad-exception-caught,import-outside-toplevel

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank-interpolated percentile of ``values`` (``pct`` in [0, 100])."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 2)
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return round(ordered[lo] + (ordered[hi] - ordered[lo]) * frac, 2)


@dataclass
class BenchResult:
    """Per-run latencies (ms) + the silence budget the bench asserts against."""

    latencies_ms: list[float] = field(default_factory=list)
    silence_window_ms: float = 700.0
    headroom: float = 0.5  # the p95 must be <= headroom * window
    warm_ms: float = 0.0

    @property
    def p50(self) -> float:
        return percentile(self.latencies_ms, 50)

    @property
    def p95(self) -> float:
        return percentile(self.latencies_ms, 95)

    @property
    def budget_ms(self) -> float:
        """The latency ceiling: a fraction (``headroom``) of the silence window."""
        return round(self.silence_window_ms * self.headroom, 2)


def fits_silence_window(result: BenchResult) -> bool:
    """True when the p95 inference latency fits within the budget (G4 assertion).

    The budget is ``headroom`` × the silence window (default: half of 700 ms =
    350 ms). Using the p95 (not the mean) guards against the occasional slow run
    that would clip a user mid-sentence. Empty results are treated as a failure
    (nothing was measured to prove it fits).
    """
    if not result.latencies_ms:
        return False
    return result.p95 <= result.budget_ms


def summarize(result: BenchResult) -> dict[str, object]:
    """A compact, JSON-able summary of the bench (latencies + the pass/fail verdict)."""
    return {
        "runs": len(result.latencies_ms),
        "warm_ms": round(result.warm_ms, 2),
        "p50_ms": result.p50,
        "p95_ms": result.p95,
        "silence_window_ms": result.silence_window_ms,
        "budget_ms": result.budget_ms,
        "fits_silence_window": fits_silence_window(result),
    }


def run_bench(
    infer: Callable[[], object],
    *,
    runs: int,
    silence_window_ms: float,
    headroom: float,
    clock: Callable[[], float] = time.perf_counter,
) -> BenchResult:
    """Time ``infer`` ``runs`` times (after one warm run) into a :class:`BenchResult`.

    ``infer`` runs one Smart-Turn inference; ``clock`` is injectable so tests drive
    it with a deterministic fake clock (no real model, no wall-clock sleeps).
    """
    result = BenchResult(silence_window_ms=silence_window_ms, headroom=headroom)
    start = clock()
    infer()  # warm: model load + first-run graph/Metal warm-up (not counted)
    result.warm_ms = (clock() - start) * 1000.0
    for _ in range(runs):
        t0 = clock()
        infer()
        result.latencies_ms.append((clock() - t0) * 1000.0)
    return result


def _build_real_infer(cfg: object) -> Callable[[], object]:
    """Build a one-inference callable over the REAL Smart Turn analyzer (on-device).

    Raises if the model / onnxruntime / feature extractor are unavailable so the
    caller reports ``skipped`` instead of benchmarking a fallback. Feeds a fixed
    8 s of synthetic 16 kHz audio (the model's window) each call.
    """
    import numpy as np

    from my_stt_tts.turn import SmartTurnAnalyzer

    analyzer = SmartTurnAnalyzer(
        getattr(cfg, "smart_turn_model_path", "models/smart-turn-v3.0.onnx"),
        silence_seconds=getattr(cfg, "vad_silence_seconds", 0.7),
        frame_seconds=512 / 16000,
        threshold=getattr(cfg, "smart_turn_threshold", 0.5),
        model_url=getattr(cfg, "smart_turn_model_url", ""),
        auto_download=getattr(cfg, "smart_turn_auto_download", False),
        expected_sha256=getattr(cfg, "smart_turn_sha256", ""),
    )
    if not analyzer._ensure_model():  # noqa: SLF001 — bench needs the loaded session
        raise RuntimeError("Smart Turn model / runtime unavailable")
    analyzer._audio = [np.zeros(8 * 16000, dtype=np.float32)]  # noqa: SLF001
    return analyzer._completion_probability  # noqa: SLF001 — time just the inference


def main(argv: list[str] | None = None) -> int:
    """Run the Smart-Turn latency bench and report pass/fail against the budget."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, default=30, help="Timed inferences (default 30).")
    parser.add_argument(
        "--headroom",
        type=float,
        default=0.5,
        help="Fraction of the silence window the p95 must fit under (default 0.5).",
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON.")
    args = parser.parse_args(argv)

    try:
        from my_stt_tts.config import Config

        cfg = Config.from_env()
        silence_window_ms = cfg.vad_silence_seconds * 1000.0
        infer = _build_real_infer(cfg)
    except Exception as exc:
        msg = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(msg) if args.json else f"skipped: {msg['reason']}")
        return 0

    result = run_bench(
        infer, runs=args.runs, silence_window_ms=silence_window_ms, headroom=args.headroom
    )
    summary = summarize(result)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"{'metric':22} {'value':>10}")
        for key, val in summary.items():
            print(f"{key:22} {val!s:>10}")
        verdict = "PASS" if summary["fits_silence_window"] else "FAIL"
        print(f"\nSmart-Turn p95 {result.p95} ms vs budget {result.budget_ms} ms -> {verdict}")
    # Non-zero exit when the model does NOT fit the silence window (so CI/bench fails).
    return 0 if summary["fits_silence_window"] else 1


if __name__ == "__main__":
    sys.exit(main())
