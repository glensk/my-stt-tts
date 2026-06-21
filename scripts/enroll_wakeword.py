#!/usr/bin/env python3
"""Enroll a CUSTOM wake word from a few of YOUR OWN clips (EfficientWord-Net's idea).

Records N short clips of the word (or reuses every clip already saved under
``debug/recordings/wake/<word>/``), mean-pools each clip's openWakeWord embedding into a
reference vector, and saves the per-clip references to the gitignored
``models/wake_embeddings/<word>.npz``. The live detector then fires on the MAX cosine
similarity of streaming audio to those references — no GPU retrain, the few-shot path
openWakeWord lacks. OR'd with openWakeWord + sherpa-KWS for that custom word; OFFICIAL words
(hey_jarvis/alexa/hey_mycroft) are never enrolled (they already fire 99-100%). Needs the
``audio`` + ``wake`` extras.

Usage:
    uv run scripts/enroll_wakeword.py <word> [--clips N] [--seconds S]
                                            [--threshold T] [--patience P]
    uv run scripts/enroll_wakeword.py <word> --from-saved   # reuse saved wake clips, no mic
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse

from _bootstrap import ensure_venv

ensure_venv(["audio", "wake"])

import numpy as np  # noqa: E402  (after the venv re-exec guarantees it's installed)


def main(argv: list[str] | None = None) -> int:
    """Record (or reuse saved) clips of a custom word and save its enrolled references."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("word", help="The custom wake word to enroll (e.g. maziko).")
    parser.add_argument("--clips", type=int, default=6, help="Number of clips to record.")
    parser.add_argument("--seconds", type=float, default=2.0, help="Max seconds per clip.")
    parser.add_argument(
        "--from-saved",
        action="store_true",
        help="Skip recording; enroll from every clip already saved under "
        "debug/recordings/wake/<word>/ (+ loose *-<word>-*.wav).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Print the suggested .env FEWSHOT_THRESHOLD line with this value (does not save).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Print the suggested .env FEWSHOT_PATIENCE line with this value (does not save).",
    )
    args = parser.parse_args(argv)

    from my_stt_tts import audio
    from my_stt_tts.config import is_official_wake_word
    from my_stt_tts.enrolled_wake import enroll_word

    if is_official_wake_word(args.word):
        print(
            f"'{args.word}' is an OFFICIAL openWakeWord word — it already fires reliably and is "
            "never enrolled (openWakeWord-only). Pick a custom word."
        )
        return 1

    clips: list[np.ndarray] | None = None
    if not args.from_saved:
        print(f"Enrolling '{args.word}' — say the word clearly for each clip.")
        recorded: list[np.ndarray] = []
        for index in range(args.clips):
            print(f"Clip {index + 1}/{args.clips} — say '{args.word}':")
            clip = audio.record_push_to_talk(16000, args.seconds, prompt="  [Enter] start/stop: ")
            if clip.size == 0:
                print("  (empty, skipped)")
                continue
            recorded.append(clip)
            # ALSO save it as training data so a later --from-saved re-enroll picks it up.
            audio.save_recording(clip, 16000, kind="wake", source="server", word=args.word)
        if not recorded:
            print("No audio captured — nothing saved.")
            return 1
        clips = recorded

    result = enroll_word(args.word, clips=clips)
    print(result["message"])
    if not result["enrolled"]:
        return 1
    print(
        f"\nThe few-shot detector is now wired for '{args.word}' (OR'd with openWakeWord + KWS).\n"
        f"It is enabled by default; tune via .env:"
    )
    thr = args.threshold if args.threshold is not None else 0.96
    pat = args.patience if args.patience is not None else 2
    print(f"  WAKE_PHRASE={args.word}")
    print(f"  FEWSHOT_THRESHOLD={thr}   # cosine 0..1; higher = stricter")
    print(f"  FEWSHOT_PATIENCE={pat}    # consecutive windows to fire; 2 = fewer false-accepts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
