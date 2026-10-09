#!/usr/bin/env python3
"""Spike S-BRIGHT: do System Events key codes 144/145 change the built-in display?

Reads the brightness of every online display through the private
DisplayServices framework (CoreDisplay as a cross-check), presses the keys via
`osascript`, measures the step size, and restores the starting value with
DisplayServicesSetBrightness. Writes a JSON verdict.

Examples:
  scripts/spikes/s_bright.py -r                 # read only
  scripts/spikes/s_bright.py -n 3 -o out.json   # 3 presses each way, write JSON
"""

from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

CG = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
DS = ctypes.CDLL("/System/Library/PrivateFrameworks/DisplayServices.framework/DisplayServices")
CD = ctypes.CDLL("/System/Library/Frameworks/CoreDisplay.framework/CoreDisplay")
CG.CGMainDisplayID.restype = ctypes.c_uint32
CG.CGDisplayIsBuiltin.restype = ctypes.c_bool
CD.CoreDisplay_Display_GetUserBrightness.restype = ctypes.c_double
DS.DisplayServicesSetBrightness.argtypes = [ctypes.c_uint32, ctypes.c_float]

KEY_UP = 144
KEY_DOWN = 145


def displays() -> list[int]:
    """Return the ids of all online displays."""
    arr = (ctypes.c_uint32 * 16)()
    count = ctypes.c_uint32(0)
    CG.CGGetOnlineDisplayList(16, arr, ctypes.byref(count))
    return [arr[i] for i in range(count.value)]


def builtin_display() -> int | None:
    """Return the built-in display id, or None."""
    return next((d for d in displays() if CG.CGDisplayIsBuiltin(d)), None)


def read(display: int) -> float | None:
    """Read brightness 0..1 via DisplayServices; None when unsupported."""
    val = ctypes.c_float(0.0)
    rc = DS.DisplayServicesGetBrightness(display, ctypes.byref(val))
    return round(val.value, 4) if rc == 0 else None


def read_coredisplay(display: int) -> float:
    """Cross-check reading via CoreDisplay."""
    return round(CD.CoreDisplay_Display_GetUserBrightness(display), 4)


def press(code: int) -> None:
    """Press a key code through System Events."""
    subprocess.run(
        ["osascript", "-e", f'tell application "System Events" to key code {code}'],
        check=True,
        capture_output=True,
    )
    time.sleep(0.6)  # the HUD animates the change


def run(presses: int) -> dict:
    """Press down then up `presses` times each, record values, restore."""
    disp = builtin_display()
    main_id = CG.CGMainDisplayID()
    result: dict = {
        "spike": "S-BRIGHT",
        "date": datetime.now(UTC).isoformat(timespec="seconds"),
        "read_method": "ctypes DisplayServices.DisplayServicesGetBrightness(display, &float)"
        " (private framework); cross-check CoreDisplay_Display_GetUserBrightness",
        "set_method": "DisplayServices.DisplayServicesSetBrightness(display, float)",
        "key_method": "osascript -e 'tell application \"System Events\" to key code N'",
        "displays": [
            {"id": d, "builtin": bool(CG.CGDisplayIsBuiltin(d)), "main": d == main_id}
            for d in displays()
        ],
        "builtin_is_main": disp == main_id,
    }
    if disp is None:
        result["verdict"] = "no built-in display"
        return result
    before = read(disp)
    result["before"] = before
    seq: list[dict] = []
    for code, label in ((KEY_DOWN, "down"), (KEY_UP, "up")):
        for _ in range(presses):
            prev = read(disp)
            press(code)
            now = read(disp)
            seq.append(
                {
                    "key": code,
                    "dir": label,
                    "from": prev,
                    "to": now,
                    "coredisplay": read_coredisplay(disp),
                }
            )
    result["presses"] = seq
    downs = [
        s["from"] - s["to"] for s in seq if s["dir"] == "down" and None not in (s["from"], s["to"])
    ]
    ups = [
        s["to"] - s["from"] for s in seq if s["dir"] == "up" and None not in (s["from"], s["to"])
    ]
    result["step_down_mean"] = round(sum(downs) / len(downs), 4) if downs else None
    result["step_up_mean"] = round(sum(ups) / len(ups), 4) if ups else None
    after_keys = read(disp)
    result["after_keys"] = after_keys
    if before is not None:
        DS.DisplayServicesSetBrightness(disp, ctypes.c_float(before))
        time.sleep(0.5)
    result["after_restore"] = read(disp)
    works_down = any(d > 0.001 for d in downs)
    works_up = any(u > 0.001 for u in ups)
    result["key_144_up_works"] = works_up
    result["key_145_down_works"] = works_down
    result["verdict"] = "pass" if works_up and works_down else "fail"
    return result


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "-r", "--read", action="store_true", help="only print current brightness per display"
    )
    ap.add_argument(
        "-n", "--presses", type=int, default=3, help="presses per direction (default 3)"
    )
    ap.add_argument("-o", "--output", type=Path, help="write the JSON result here")
    args = ap.parse_args()
    if args.read:
        for d in displays():
            print(
                json.dumps(
                    {"id": d, "builtin": bool(CG.CGDisplayIsBuiltin(d)), "brightness": read(d)}
                )
            )
        return 0
    res = run(args.presses)
    text = json.dumps(res, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    return 0 if res.get("verdict") == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
