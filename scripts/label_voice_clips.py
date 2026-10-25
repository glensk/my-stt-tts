#!/usr/bin/env -S uv run --no-sync --project /Users/albert/obsidian/42-Git/infra/my-stt-tts python
"""Label the in-call clips mac-voice kept: was that you ("albert") or someone else ("other")?

During calls mac-voice keeps speech the voice check refused, speech the agent ignored as not
meant for it, and a sample of accepted speech (``debug/recordings/interference/``, see
``my_stt_tts.interference``). A label turns a clip into training data for
``scripts/calibrate_voice.py``: ``albert`` clips are positives, ``other`` clips negatives
(no names — anyone who is not Albert is ``other``). Labelled clips are never auto-deleted;
unlabelled ones go after 90 days (``MAC_VOICE_INTERFERENCE_DAYS``).

Usage:
    scripts/label_voice_clips.py -p           # play each unlabelled clip and label it
    scripts/label_voice_clips.py -l           # list unlabelled clips
    scripts/label_voice_clips.py -l -A        # list every clip with its label
    scripts/label_voice_clips.py -p -d skipped   # only clips the agent ignored
    scripts/label_voice_clips.py -s 20261025-2012 other   # label one clip (id or unique prefix)
    scripts/label_voice_clips.py -s 20261025-2012 none    # remove its label
    scripts/label_voice_clips.py -x 20261025-2012         # delete one clip
    scripts/label_voice_clips.py -e           # delete unlabelled clips past the expiry now

While playing (-p): a = albert, o = other, r = replay, s = skip, d = delete, q = quit.

Exit codes: 0 done, 1 no such clip, 2 usage error.
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from _bootstrap import ensure_venv

ensure_venv([])

# pylint: disable=wrong-import-position,wrong-import-order  # after the venv re-exec
from my_stt_tts import interference as itf  # noqa: E402, RUF100
from my_stt_tts import voice_profiles as vp  # noqa: E402, RUF100

# pylint: enable=wrong-import-position,wrong-import-order

KEYS = {"a": "albert", "o": "other"}
PROMPT = "  a = albert, o = other, r = replay, s = skip, d = delete, q = quit: "


def _afplay(path: Path) -> None:
    subprocess.run(["/usr/bin/afplay", str(path)], check=False)


@dataclass
class Deps:
    """The outside world (tests replace it)."""

    play: Callable[[Path], None] = _afplay
    ask: Callable[[str], str] = input
    out: Callable[[str], None] = print


def describe(clip: itf.Clip) -> str:
    """``20261025-201200-skipped-ab12cd  skipped        0.31 call  1.8 s  -``"""
    score = "-" if clip.score is None else f"{clip.score:.2f}"
    profile = clip.meta.get("profile") or "-"
    return (
        f"{clip.id:<38} {clip.decision:<13} {score:>5} {profile:<4} {clip.duration:4.1f} s"
        f"  {clip.label or '-'}"
    )


def select(store: itf.ClipStore, args: argparse.Namespace) -> list[itf.Clip]:
    """The clips ``-l`` / ``-p`` work on: unlabelled unless ``-A``, optional decision filter."""
    clips = store.clips()
    if not args.all:
        clips = [c for c in clips if c.label is None]
    if args.decision:
        clips = [c for c in clips if c.decision == args.decision]
    return clips


def summary(store: itf.ClipStore) -> str:
    """``12 clips: albert 3, other 4, unlabelled 5``."""
    clips = store.clips()
    counts = Counter(c.label or "unlabelled" for c in clips)
    parts = [f"{name} {counts[name]}" for name in (*itf.LABELS, "unlabelled") if counts[name]]
    return f"{len(clips)} clip(s)" + (": " + ", ".join(parts) if parts else "")


def play_and_label(store: itf.ClipStore, clips: list[itf.Clip], deps: Deps) -> None:
    """Play each clip and ask for its label."""
    for number, clip in enumerate(clips, 1):
        deps.out(f"[{number}/{len(clips)}] {describe(clip)}")
        deps.play(clip.wav)
        while True:
            answer = deps.ask(PROMPT).strip().lower()
            if answer in KEYS:
                store.set_label(clip, KEYS[answer])
                deps.out(f"  ✅ {KEYS[answer]}")
                break
            if answer == "r":
                deps.play(clip.wav)
                continue
            if answer == "d":
                store.delete(clip)
                deps.out("  ✅ deleted")
                break
            if answer == "q":
                return
            if answer in ("s", ""):
                break
            deps.out("  ❌ type a, o, r, s, d or q")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("-l", "--list", action="store_true", help="list clips (default action)")
    mode.add_argument("-p", "--play", action="store_true", help="play and label clips")
    mode.add_argument(
        "-s", "--set", nargs=2, metavar=("ID", "LABEL"), help="label one clip: albert|other|none"
    )
    mode.add_argument("-x", "--delete", metavar="ID", help="delete one clip")
    mode.add_argument(
        "-e", "--expire", action="store_true", help="delete unlabelled clips past the expiry"
    )
    parser.add_argument("-A", "--all", action="store_true", help="-l / -p: include labelled clips")
    parser.add_argument(
        "-d", "--decision", choices=itf.DECISIONS, help="-l / -p: only clips with this decision"
    )
    parser.add_argument(
        "-D", "--dir", type=Path, default=itf.ROOT, help="clip folder (default: %(default)s)"
    )
    return parser


def run(args: argparse.Namespace, deps: Deps) -> int:
    """Do what the arguments ask; the exit code."""
    store = itf.ClipStore(args.dir)
    if args.set or args.delete:
        ref = args.set[0] if args.set else args.delete
        clip = store.find(ref)
        if clip is None:
            deps.out(f"❌ no clip (or more than one) matches {ref!r}")
            return 1
        if args.delete:
            store.delete(clip)
            deps.out(f"✅ deleted {clip.id}")
            return 0
        label = args.set[1].lower()
        if label not in (*itf.LABELS, "none"):
            deps.out(f"❌ label must be albert, other or none, not {label!r}")
            return 2
        store.set_label(clip, None if label == "none" else label)
        deps.out(f"✅ {clip.id}: {label}")
        return 0
    if args.expire:
        days = vp.env_float(itf.DAYS_ENV, itf.DEFAULT_DAYS)
        gone = store.expire(days)
        deps.out(f"✅ deleted {gone} unlabelled clip(s) older than {days:g} days")
        return 0
    clips = select(store, args)
    if args.play:
        if not clips:
            deps.out("✅ nothing to label")
        else:
            play_and_label(store, clips, deps)
    else:
        for clip in clips:
            deps.out(describe(clip))
    deps.out(summary(store))
    return 0


def main(argv: list[str] | None = None, deps: Deps | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        return run(args, deps or Deps())
    except KeyboardInterrupt:
        print("\n✅ stopped — labels set so far are saved", file=sys.stderr)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
