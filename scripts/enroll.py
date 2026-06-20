#!/usr/bin/env python3
"""Enroll a household member's voice into ONE multi-language speaker-ID centroid.

Records several short clips, averages their ECAPA embeddings into one
L2-normalized centroid, and saves it to ``<out>/<name>.npy`` (gitignored). The
prompt CYCLES a language hint across the clips ("Clip 3/6 — say it in French")
rotating through the languages the household uses (default DE/EN/FR) so a SINGLE
profile spans every language the person speaks — profiles are NOT split by
language. Re-enroll children every few months. Needs the ``audio`` and ``speaker``
extras.

Usage:
    uv run scripts/enroll.py <name> [--clips N] [--seconds S] [--out DIR]
                                    [--languages de,en,fr]
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_venv

ensure_venv(["audio", "speaker"])

import numpy as np  # noqa: E402  (after the venv re-exec guarantees it's installed)

# Default household languages cycled across the enrollment clips (German, English,
# French) so one centroid covers a multilingual speaker. Override with --languages.
DEFAULT_LANGUAGES = ("de", "en", "fr")

# Friendly names for the prompt (falls back to the upper-cased code when unknown).
_LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "fr": "French",
    "it": "Italian",
    "es": "Spanish",
    "rm": "Romansh",
}


def parse_languages(raw: str) -> list[str]:
    """Parse a ``--languages de,en,fr`` string into a clean lower-cased code list.

    Splits on commas, strips whitespace, lower-cases, and drops blanks while
    preserving order + de-duplicating. Falls back to :data:`DEFAULT_LANGUAGES` when
    nothing usable is given so the cycle is never empty.
    """
    seen: list[str] = []
    for part in str(raw or "").split(","):
        code = part.strip().lower()
        if code and code not in seen:
            seen.append(code)
    return seen or list(DEFAULT_LANGUAGES)


def language_name(code: str) -> str:
    """Human name for a language ``code`` (e.g. ``"de"`` -> ``"German"``)."""
    return _LANGUAGE_NAMES.get(code, code.upper())


def clip_prompt(index: int, total: int, languages: list[str]) -> str:
    """The per-clip prompt with a CYCLED language hint (e.g. "Clip 3/6 — say it in French")."""
    lang = languages[index % len(languages)]
    return f"Clip {index + 1}/{total} — say it in {language_name(lang)}:"


def main(argv: list[str] | None = None) -> int:
    """Record enrollment clips (cycling a language hint) and save the speaker centroid."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("name", help="Person's name (used as the profile filename).")
    parser.add_argument("--clips", type=int, default=6, help="Number of clips to record.")
    parser.add_argument("--seconds", type=float, default=5.0, help="Max seconds per clip.")
    parser.add_argument("--out", default="enroll", help="Output directory for profiles.")
    parser.add_argument(
        "--languages",
        default=",".join(DEFAULT_LANGUAGES),
        help="Comma-separated language codes to CYCLE across the clips (default de,en,fr). "
        "One profile spans them all — profiles are NOT split by language.",
    )
    args = parser.parse_args(argv)
    languages = parse_languages(args.languages)

    from my_stt_tts import audio
    from my_stt_tts.speaker_id import EcapaEmbedder

    print(f"Enrolling '{args.name}' across {len(languages)} languages: {', '.join(languages)}")
    embedder = EcapaEmbedder()
    embeddings: list[np.ndarray] = []
    for index in range(args.clips):
        print(clip_prompt(index, args.clips, languages))
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
    print(
        f"Saved centroid -> {path} (dim {centroid.shape[0]}, from {len(embeddings)} clips "
        f"across {', '.join(languages)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
