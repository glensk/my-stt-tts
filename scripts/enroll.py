#!/usr/bin/env python3
"""Enroll a household member's voice into a speaker-ID centroid.

Records several short clips, averages their ECAPA embeddings into one
L2-normalized centroid, and saves it to ``<out>/<name>.npy`` (gitignored).
Record clips in each language the person uses; re-enroll children every few
months. Needs the ``audio`` and ``speaker`` extras.

Usage:
    uv run scripts/enroll.py <name> [--clips N] [--seconds S] [--out DIR]
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    """Record enrollment clips and save the speaker centroid."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="Person's name (used as the profile filename).")
    parser.add_argument("--clips", type=int, default=6, help="Number of clips to record.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Max seconds per clip.")
    parser.add_argument("--out", default="enroll", help="Output directory for profiles.")
    args = parser.parse_args(argv)

    from my_stt_tts import audio
    from my_stt_tts.speaker_id import EcapaEmbedder

    embedder = EcapaEmbedder()
    embeddings: list[np.ndarray] = []
    for index in range(args.clips):
        print(f"Clip {index + 1}/{args.clips}:")
        clip = audio.record_push_to_talk(16000, args.seconds, prompt="  [Enter] start/stop: ")
        if clip.size == 0:
            print("  (empty, skipped)")
            continue
        embeddings.append(embedder.embed(clip, 16000))

    if not embeddings:
        print("No audio captured — nothing saved.")
        return 1

    centroid = np.mean(np.stack(embeddings), axis=0)
    centroid = centroid / (float(np.linalg.norm(centroid)) or 1.0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name}.npy"
    np.save(path, centroid.astype(np.float32))
    print(f"Saved centroid -> {path} (dim {centroid.shape[0]}, from {len(embeddings)} clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
