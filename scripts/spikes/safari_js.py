#!/usr/bin/env python3
"""Drive Safari windows by stable window id for spike S-MEDIA.

JavaScript and URLs travel as osascript argv, never through shell
interpolation. Only windows this tool opened should be targeted.

Examples:
  scripts/spikes/safari_js.py -n https://www.youtube.com     # open new window, print its id
  scripts/spikes/safari_js.py -w 1234 -e 'location.href'    # evaluate an expression
  scripts/spikes/safari_js.py -w 1234 -f snippet.js         # evaluate a file
  scripts/spikes/safari_js.py -w 1234 -u https://example.org # navigate that window
  scripts/spikes/safari_js.py -w 1234 -b                    # window bounds + front state
  scripts/spikes/safari_js.py -w 1234 -c                    # close that window
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

NEW_WINDOW = """on run argv
tell application "Safari"
  make new document with properties {URL:(item 1 of argv)}
  return id of front window
end tell
end run"""

RUN_JS = """on run argv
tell application "Safari" to do JavaScript (item 1 of argv) in current tab of window id ((item 2 of argv) as integer)
end run"""

SET_URL = """on run argv
tell application "Safari" to set URL of current tab of window id ((item 2 of argv) as integer) to (item 1 of argv)
end run"""

CLOSE = """on run argv
tell application "Safari" to close window id ((item 1 of argv) as integer)
end run"""

BOUNDS = """on run argv
tell application "Safari"
  set w to window id ((item 1 of argv) as integer)
  set b to bounds of w
  set fi to (index of w)
  return (item 1 of b as text) & "," & (item 2 of b as text) & "," & (item 3 of b as text) & "," & (item 4 of b as text) & ",index=" & fi & ",frontmost=" & (frontmost as text)
end tell
end run"""


def osa(script: str, *argv: str, timeout: float = 30) -> str:
    """Run an AppleScript with argv; return stdout or raise on error."""
    proc = subprocess.run(
        ["osascript", "-e", script, *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip())
    return proc.stdout.rstrip("\n")


def new_window(url: str) -> int:
    """Open a new Safari window on url, return its id."""
    return int(osa(NEW_WINDOW, url))


def js(window: int, code: str) -> str:
    """Evaluate JavaScript in the current tab of a window."""
    return osa(RUN_JS, code, str(window))


def set_url(window: int, url: str) -> None:
    """Navigate the current tab of a window."""
    osa(SET_URL, url, str(window))


def close(window: int) -> None:
    """Close a window by id."""
    osa(CLOSE, str(window))


def bounds(window: int) -> str:
    """Return bounds, index and Safari frontmost state."""
    return osa(BOUNDS, str(window))


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-n", "--new", metavar="URL", help="open a new window on URL and print its id")
    ap.add_argument("-w", "--window", type=int, help="target window id")
    ap.add_argument("-e", "--expr", help="JavaScript expression to evaluate")
    ap.add_argument("-f", "--file", type=Path, help="JavaScript file to evaluate")
    ap.add_argument("-u", "--url", help="navigate the window to URL")
    ap.add_argument(
        "-b", "--bounds", action="store_true", help="print window bounds/index/frontmost"
    )
    ap.add_argument("-c", "--close", action="store_true", help="close the window")
    args = ap.parse_args()
    if args.new:
        print(new_window(args.new))
        return 0
    if args.window is None:
        ap.error("-w/--window is required unless -n is given")
    if args.url:
        set_url(args.window, args.url)
    if args.expr:
        print(js(args.window, args.expr))
    if args.file:
        print(js(args.window, args.file.read_text()))
    if args.bounds:
        print(bounds(args.window))
    if args.close:
        close(args.window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
