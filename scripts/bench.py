#!/usr/bin/env python3
"""Measure per-stage latency of the voice pipeline on THIS machine.

Times whichever backends are installed and skips the rest, so you can run it
after installing just some extras (e.g. ``uv sync --extra stt --extra tts``).
Use it to validate the PLAN.md latency budget against your actual M1.

Usage:
    uv run scripts/bench.py [--text TEXT] [--llm]
"""
# pylint: disable=broad-exception-caught,import-outside-toplevel

from __future__ import annotations

import argparse
import shutil
import subprocess
import time

from _bootstrap import ensure_venv

ensure_venv(["all"])  # bench whichever backends are installed in the full venv

import numpy as np  # noqa: E402  (after the venv re-exec guarantees it's installed)


def _time_ms(func) -> float:  # noqa: ANN001
    start = time.perf_counter()
    func()
    return (time.perf_counter() - start) * 1000.0


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and print a per-stage table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--text", default="Hello, this is a latency benchmark.")
    parser.add_argument("--llm", action="store_true", help="Also time LLM time-to-first-token.")
    args = parser.parse_args(argv)

    rows: list[tuple[str, float, str]] = []

    from my_stt_tts import chimes

    rows.append(("chime gen", _time_ms(chimes.chime_listening), "ok"))

    if shutil.which("say"):
        rows.append(
            (
                "tts: macOS say",
                _time_ms(lambda: subprocess.run(["say", args.text], check=False)),
                "ok",
            )
        )

    try:
        from my_stt_tts.stt import ParakeetSTT

        stt = ParakeetSTT()
        clip = np.zeros(int(16000 * 3), dtype=np.float32)
        stt.transcribe(clip)  # warm: load model + Metal kernels (not timed)
        rows.append(("stt: parakeet 3s", _time_ms(lambda: stt.transcribe(clip)), "ok"))
    except Exception as exc:
        rows.append(("stt: parakeet 3s", 0.0, f"skipped: {type(exc).__name__}"))

    if args.llm:
        try:
            from my_stt_tts.brain import Brain
            from my_stt_tts.config import Config

            cfg = Config.from_env()
            cfg.validate()
            brain = Brain(cfg)
            start = time.perf_counter()
            ttft = 0.0
            for _ in brain.stream("Say hi in one word."):
                ttft = (time.perf_counter() - start) * 1000.0
                break
            rows.append(("llm TTFT", ttft, "ok"))
        except Exception as exc:
            rows.append(("llm TTFT", 0.0, f"skipped: {type(exc).__name__}"))

    print(f"{'stage':22} {'ms':>9}  status")
    for name, ms, status in rows:
        print(f"{name:22} {ms:9.1f}  {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
