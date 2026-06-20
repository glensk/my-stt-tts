#!/usr/bin/env python3
"""Calibrate the speaker-ID threshold from enrolled profiles + labeled test clips.

Loads per-person centroids from the enroll dir (``<name>.npy``, made by
``enroll.py``) and held-out test clips laid out as::

    tests_audio/alice/*.wav
    tests_audio/bob/*.wav
    tests_audio/unknown/*.wav     # guest / impostor clips (optional but recommended)

It embeds the test clips (ECAPA), sweeps the match threshold, prints an
accuracy / impostor-accept table, and recommends a ``speaker_threshold`` that
rejects strangers. Needs the ``audio`` + ``speaker`` extras.

Usage:
    uv run scripts/calibrate.py [--enroll enroll] [--tests tests_audio] [--margin 0.06]
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import wave
from pathlib import Path

from _bootstrap import ensure_venv

ensure_venv(["audio", "speaker"])

import numpy as np  # noqa: E402  (after the venv re-exec guarantees it's installed)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return audio, sample_rate


def main(argv: list[str] | None = None) -> int:
    """Run the calibration sweep and print the recommended threshold."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--enroll", default="enroll", help="Dir of <name>.npy centroids.")
    parser.add_argument("--tests", default="tests_audio", help="Dir of <name>/*.wav test clips.")
    parser.add_argument("--margin", type=float, default=0.06)
    args = parser.parse_args(argv)

    from my_stt_tts.speaker_id import EcapaEmbedder, calibrate_threshold

    enroll_dir = Path(args.enroll)
    centroids = {p.stem: np.load(p) for p in enroll_dir.glob("*.npy")}
    if not centroids:
        print(f"No centroids in {enroll_dir} — run scripts/enroll.py first.")
        return 1

    embedder = EcapaEmbedder()
    labeled: dict[str, list[np.ndarray]] = {}
    for speaker_dir in sorted(Path(args.tests).glob("*")):
        if not speaker_dir.is_dir():
            continue
        clips = []
        for wav in sorted(speaker_dir.glob("*.wav")):
            audio, sample_rate = _read_wav(wav)
            clips.append(embedder.embed(audio, sample_rate))
        if clips:
            labeled[speaker_dir.name] = clips
    if not labeled:
        print(f"No test clips under {args.tests}/<name>/*.wav")
        return 1

    best, rows = calibrate_threshold(centroids, labeled, margin=args.margin)
    print(f"{'threshold':>10} {'accuracy':>9} {'impostor-accept':>16}")
    for thr, acc, far in rows:
        mark = "  <- recommended" if thr == best else ""
        print(f"{thr:>10.2f} {acc:>9.3f} {far:>16.3f}{mark}")
    print(f"\nRecommended: speaker_threshold = {best}  (set SPEAKER_THRESHOLD or edit config).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
