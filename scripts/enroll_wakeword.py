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

Recording is hands-free: after each prompt just say the word; a voice detector notices
the start and the end (no Enter key). Clips ACCUMULATE: every run adds its clips to the
saved ones and re-enrolls from all of them, so several people can enroll one after the
other (``-w NAME`` tags whose clips they are).

Usage:
    uv run scripts/enroll_wakeword.py <word> [-n N] [-w NAME] [-s S]
                                            [--threshold T] [--patience P]
    uv run scripts/enroll_wakeword.py "voice on" -n 8 -w albert   # 8 clips, tagged albert
    uv run scripts/enroll_wakeword.py "voice on" -w anna          # add another person
    uv run scripts/enroll_wakeword.py <word> --from-saved   # reuse saved wake clips, no mic
    uv run scripts/enroll_wakeword.py <word> -p             # old Enter-to-start/stop mode
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import re
import time
from typing import Any

from _bootstrap import ensure_venv

ensure_venv(["audio", "wake"])


def main(argv: list[str] | None = None) -> int:
    """Record (or reuse saved) clips of a custom word and save its enrolled references."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("word", help="The custom wake word to enroll (e.g. maziko).")
    parser.add_argument("-n", "--clips", type=int, default=6, help="Number of clips to record.")
    parser.add_argument("-s", "--seconds", type=float, default=3.0, help="Max seconds per clip.")
    parser.add_argument(
        "-w", "--who", default="", help="Whose voice this is (tags the saved clips, e.g. anna)."
    )
    parser.add_argument(
        "-p", "--push-to-talk", action="store_true", help="Press Enter to start/stop each clip."
    )
    parser.add_argument(
        "-f",
        "--from-saved",
        action="store_true",
        help="Skip recording; enroll from every clip already saved under "
        "debug/recordings/wake/<word>/ (+ loose *-<word>-*.wav).",
    )
    parser.add_argument(
        "-T",
        "--threshold",
        type=float,
        default=None,
        help="Print the suggested .env FEWSHOT_THRESHOLD line with this value (does not save).",
    )
    parser.add_argument(
        "-P",
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

    if not args.from_saved:
        recorded = _record_clips(args, audio)
        if not recorded:
            print("❌ No audio captured — nothing saved.")
            return 1
        print(f"✅ recorded {recorded}× '{args.word}' — enrolling from all saved clips …")
        _update_voice_profile(args.who)

    # Clips accumulate: always enroll from EVERY saved clip of the word (all speakers).
    result = enroll_word(args.word, clips=None)
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


def _update_voice_profile(who: str) -> None:
    """Refresh who's voice profile (mac-voice: only enrolled voices may start it)."""
    name = re.sub(r"[^a-z0-9]+", "", who.lower())
    if not name:
        return
    from my_stt_tts.voice_gate import build_profile

    path, used, found = build_profile(name)
    if path is None:
        print(f"⚠️  voice profile for {name}: {found} clips so far, need 3")
    else:
        print(f"🗣️  voice profile for {name} updated from {used}/{found} clips → {path}")


def _record_clips(args: argparse.Namespace, audio: Any) -> int:
    """Record ``args.clips`` clips (hands-free unless ``-p``); save each; return the count."""
    who = re.sub(r"[^a-z0-9]+", "", args.who.lower())
    source = f"enroll_{who}" if who else "server"
    voice = f" ({args.who})" if args.who else ""
    print(f"🎙️  Enrolling '{args.word}'{voice} — say it once after each prompt, naturally.")
    vad = endpointer = None
    if not args.push_to_talk:
        from my_stt_tts.vad import SilenceEndpointer, SileroVad

        vad = SileroVad(16000, 0.3)
        endpointer = SilenceEndpointer(0.6, frame_seconds=512 / 16000)
    done, misses = 0, 0
    while done < args.clips and misses < 3:
        print(f"👉 [{done + 1}/{args.clips}] say '{args.word}' …", flush=True)
        if args.push_to_talk:
            clip = audio.record_push_to_talk(16000, args.seconds, prompt="  [Enter] start/stop: ")
        else:
            clip = audio.record_until_silence(16000, vad, endpointer, max_seconds=args.seconds)
        if clip.size < 16000 * 0.25:
            misses += 1
            print(f"   ⚠️  nothing heard — try again ({3 - misses} tries left)")
            continue
        misses = 0
        done += 1
        audio.save_recording(clip, 16000, kind="wake", source=source, word=args.word)
        print(f"   ✅ recorded {done}× '{args.word}' ({clip.size / 16000:.1f} s)")
        time.sleep(0.4)  # a beat before the next prompt
    return done


if __name__ == "__main__":
    raise SystemExit(main())
