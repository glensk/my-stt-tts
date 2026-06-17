#!/usr/bin/env python3
"""Test a trained openWakeWord model against a WAV file (verify it fires).

Slides 80 ms frames through the model and reports where (if anywhere) the wake
word fires. Use it after training to sanity-check `wakewords/maziko.onnx` on a
recording of yourself saying the phrase. Expects 16 kHz mono WAV.

Usage:
    uv run scripts/test_wakeword.py <model.onnx> <audio.wav> [--threshold 0.5]
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import wave

import numpy as np


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as handle:
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate


def main(argv: list[str] | None = None) -> int:
    """Run the wake-word model over the WAV and print whether it fired."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", help="Path to the trained .onnx wake-word model.")
    parser.add_argument("wav", help="Path to a 16 kHz mono WAV to test against.")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    from my_stt_tts.wake import WakeWord

    audio, sample_rate = _read_wav(args.wav)
    if sample_rate != 16000:
        print(f"warning: expected 16 kHz mono, got {sample_rate} Hz")
    wake = WakeWord(args.model, args.threshold)
    step = 1280  # 80 ms at 16 kHz
    fired = False
    for i in range(0, max(0, len(audio) - step), step):
        if wake.detect(audio[i : i + step]):
            print(f"WAKE fired at ~{i / sample_rate:.2f}s")
            fired = True
    print("RESULT:", "fired ✓" if fired else "did NOT fire ✗")
    return 0 if fired else 1


if __name__ == "__main__":
    raise SystemExit(main())
