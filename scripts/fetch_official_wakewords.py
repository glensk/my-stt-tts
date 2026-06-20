#!/usr/bin/env python3
"""Place openWakeWord's official pre-trained wake-word models into ``wakewords/``.

openWakeWord ships several *extensively-trained, official* wake-word models
(``alexa``, ``hey_jarvis``, ``hey_mycroft``, ``hey_marvin`` — depending on the
installed version). These are trained on far more data than the small self-trained
models in this repo, so they are the **recommended reliable choice** (marked
``green`` in the wake-word reliability tiers — see ``wakewords/WAKEWORD.md``).

This script obtains them and copies the ``.onnx`` into ``wakewords/<name>.onnx``
under the repo's naming convention (so a user just picks the name). It is
version-tolerant:

* If ``openwakeword.utils.download_models()`` exists (newer releases), it is
  called first to ensure the official models are downloaded.
* Either way, the concrete on-disk model paths are read from the installed
  package (``openwakeword.get_pretrained_model_paths()`` / ``openwakeword.models``)
  and copied into ``wakewords/``.

The shared melspectrogram / embedding feature models openWakeWord needs at
runtime live inside the installed package's ``resources/models/`` and are loaded
from there automatically — they do NOT need to be copied into ``wakewords/`` (a
model loaded from an arbitrary path still finds them). Only the per-wake-word
classifier ``.onnx`` is shipped here.

Usage:
    uv run scripts/fetch_official_wakewords.py [--dest wakewords] [--list] [-h]
"""
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# The official wake-PHRASE models we ship (the ones openWakeWord registers via
# get_pretrained_model_paths(), excluding the non-wake ``timer`` / ``weather``
# intent helpers). Maps the upstream model key -> the clean ``wakewords/<stem>.onnx``
# we write. ``hey_marvin`` ships in some wheels as a file but is not a registered
# pretrained wake model, so it is intentionally not in this set.
OFFICIAL_WAKE_KEYS = {
    "alexa": "alexa",
    "hey_jarvis": "hey_jarvis",
    "hey_mycroft": "hey_mycroft",
}

# A non-wake-word helper-model substring we never ship as a selectable wake word.
_NON_WAKE = ("melspectrogram", "embedding", "silero", "timer", "weather", "vad")


def _model_paths() -> dict[str, Path]:
    """Concrete on-disk paths of the installed official models, keyed by stem.

    Prefers ``openwakeword.get_pretrained_model_paths()``; falls back to the
    ``openwakeword.models`` mapping. Runs ``download_models()`` first when present
    so a fresh install fetches the official weights before we read their paths.
    """
    import openwakeword

    utils = getattr(openwakeword, "utils", None)
    downloader = getattr(openwakeword, "download_models", None) or getattr(
        utils, "download_models", None
    )
    if callable(downloader):
        try:
            downloader()  # pylint: disable=not-callable  # guarded by callable() above
        except Exception as exc:  # noqa: BLE001 — best-effort; bundled weights still work
            print(f"note: download_models() failed ({exc}); using bundled weights")

    found: dict[str, Path] = {}
    getter = getattr(openwakeword, "get_pretrained_model_paths", None)
    if callable(getter):
        for raw in getter() or []:
            path = Path(str(raw))
            stem = re.sub(r"_v\d.*$", "", path.stem)  # hey_jarvis_v0.1 -> hey_jarvis
            if any(tok in path.stem.lower() for tok in _NON_WAKE):
                continue
            found[stem] = path
    models = getattr(openwakeword, "models", {}) or {}
    for key, info in models.items():
        if any(tok in str(key).lower() for tok in _NON_WAKE):
            continue
        mp = info.get("model_path") if isinstance(info, dict) else None
        if mp:
            found.setdefault(key, Path(str(mp)))
    return found


def fetch(dest: Path) -> list[str]:
    """Copy the official wake-word models into ``dest``. Returns the stems written."""
    paths = _model_paths()
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for upstream_key, out_stem in OFFICIAL_WAKE_KEYS.items():
        src = paths.get(upstream_key)
        if src is None or not src.is_file():
            print(f"skip {upstream_key}: not provided by this openwakeword version")
            continue
        out = dest / f"{out_stem}.onnx"
        shutil.copyfile(src, out)
        size_kb = out.stat().st_size / 1024
        print(f"wrote {out}  ({size_kb:.0f} KB)  <- {src.name}")
        written.append(out_stem)
    return written


def main(argv: list[str] | None = None) -> int:
    """Fetch official openWakeWord models into the destination dir."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dest",
        default="wakewords",
        help="Destination directory for the .onnx models (default: wakewords).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list the official models the installed openwakeword provides; do not copy.",
    )
    args = parser.parse_args(argv)

    if args.list:
        for stem, path in sorted(_model_paths().items()):
            print(f"{stem:16s} {path}")
        return 0

    written = fetch(Path(args.dest))
    if not written:
        print("no official models were written (check the openwakeword install)")
        return 1
    print(f"done: {len(written)} official model(s) -> {args.dest}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
