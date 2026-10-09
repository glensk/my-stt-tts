#!/usr/bin/env -S uv run --no-sync --project /Users/albert/obsidian/42-Git/infra/my-stt-tts python
"""Prototype Mac broker: a local stdio MCP server for the operator spike (S-OPERATOR).

Phase-0 prototype of ``mac_broker.py`` (PLAN_claude-bridge.md section 7.4). It
exposes a few primitives to a ``claude -p`` operator:

    observe_screen(app?)            screenshot of the front window (or screen) as an
                                    MCP image block + obs_id + sequence number
    open_app(name)                  ``open -a <name>``
    key(combo, app)                 System Events keystroke / key code, argv-safe
    type_text(app, text)            System Events keystroke of literal text, argv-safe
    list_ui(app)                    truncated System Events UI element dump
    finish(status, summary, evidence_obs_id)
                                    final result; ``done`` needs an observation taken
                                    AFTER the last action, else ``unknown_partial``

Every call is logged (tool name + sha256 of the canonical args, never content) to
``<call-dir>/broker_calls.jsonl`` (0600); the last finish() is written to
``<call-dir>/finish.json`` (0600). No risk table here (that is Phase 5).

Usage:
    broker_proto.py -c CALL_DIR [-m MAX_PX] [-k KEEP_DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.utilities.types import Image
from mcp.types import TextContent

# AppleScript bodies take every user-supplied value through ``argv`` so nothing is
# ever spliced into script source (no quoting/injection issues).

#: Shared AppleScript tail: return the ``out`` list as linefeed-separated text.
AS_RETURN_OUT_LINES = """    set AppleScript's text item delimiters to linefeed
    return out as text
end run
"""

_AS_FRONT_BOUNDS = """
on run argv
    tell application "System Events"
        if (count of argv) > 0 then
            set p to first process whose name is (item 1 of argv)
        else
            set p to first process whose frontmost is true
        end if
        set n to name of p
        tell p
            if (count of windows) = 0 then return n & "|none"
            set {x, y} to position of window 1
            set {w, h} to size of window 1
        end tell
    end tell
    return n & "|" & x & "," & y & "," & w & "," & h
end run
"""

_AS_KEY = """
on run argv
    set appName to item 1 of argv
    set k to item 2 of argv
    set isCode to item 3 of argv
    set modNames to {}
    if (count of argv) > 3 then set modNames to items 4 thru -1 of argv
    if appName is not "" then tell application appName to activate
    delay 0.15
    set mods to {}
    repeat with m in modNames
        set m to m as text
        if m is "cmd" then set end of mods to command down
        if m is "shift" then set end of mods to shift down
        if m is "alt" then set end of mods to option down
        if m is "ctrl" then set end of mods to control down
    end repeat
    tell application "System Events"
        if isCode is "1" then
            key code (k as integer) using mods
        else
            keystroke k using mods
        end if
    end tell
end run
"""

_AS_TYPE = """
on run argv
    set appName to item 1 of argv
    set t to item 2 of argv
    if appName is not "" then tell application appName to activate
    delay 0.15
    tell application "System Events" to keystroke t
end run
"""

_AS_LIST_UI = (
    """
on run argv
    set appName to item 1 of argv
    set maxItems to (item 2 of argv) as integer
    set out to {}
    tell application "System Events"
        tell process appName
            if (count of windows) = 0 then return "no windows"
            set els to entire contents of window 1
        end tell
        set i to 0
        repeat with e in els
            set i to i + 1
            if i > maxItems then exit repeat
            set r to ""
            set nm to ""
            set ds to ""
            set v to ""
            try
                set r to role of e
            end try
            try
                set nm to name of e as text
            end try
            try
                set ds to description of e as text
            end try
            try
                set v to value of e as text
            end try
            if nm is "missing value" then set nm to ""
            if ds is "missing value" then set ds to ""
            if v is "missing value" then set v to ""
            set end of out to r & " | " & nm & " | " & ds & " | " & v
        end repeat
    end tell
"""
    + AS_RETURN_OUT_LINES
)

_NAMED_KEYS = {
    "return": 36,
    "enter": 76,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "escape": 53,
    "esc": 53,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
}
_MODS = {"cmd": "cmd", "command": "cmd", "shift": "shift", "alt": "alt", "option": "alt"}
_MODS |= {"opt": "alt", "ctrl": "ctrl", "control": "ctrl"}
_LIST_UI_MAX_CHARS = 6000
_ERRS = (RuntimeError, ValueError, OSError, subprocess.SubprocessError)


def run_osascript(script: str, *args: str, timeout: float) -> subprocess.CompletedProcess[str]:
    """``osascript -e <script> -- <args…>`` (values via argv only); never raises on exit code."""
    return subprocess.run(
        ["osascript", "-e", script, "--", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _osascript(script: str, *args: str, timeout: float = 15.0) -> str:
    proc = run_osascript(script, *args, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"osascript failed: {proc.stderr.strip()[:300]}")
    return proc.stdout.strip()


class CaptureOptions(NamedTuple):
    """Screenshot settings: max edge in px and the optional debug keep directory."""

    max_px: int
    keep_dir: Path | None


class BrokerState:
    """Sequence counter, observation registry, last-action marker and the call log."""

    def __init__(self, call_dir: Path, max_px: int, keep_dir: Path | None) -> None:
        self.call_dir = call_dir
        self.capture = CaptureOptions(max_px, keep_dir)
        self.seq = 0
        self.last_action_seq = 0
        self.observations: dict[str, int] = {}
        self.t0 = time.monotonic()
        self.finishes: list[dict[str, Any]] = []

    @property
    def log_path(self) -> Path:
        return self.call_dir / "broker_calls.jsonl"

    @property
    def finish_path(self) -> Path:
        return self.call_dir / "finish.json"

    def next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def action_done(self, seq: int) -> None:
        """Mark *seq* as the latest action (finish() evidence must be newer)."""
        self.last_action_seq = seq

    def log(self, tool: str, args: dict[str, Any], seq: int, t_start: float, **extra: Any) -> None:
        canon = json.dumps(args, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        rec = {
            "seq": seq,
            "tool": tool,
            "args_sha256": hashlib.sha256(canon.encode()).hexdigest(),
            "t_start_s": round(t_start - self.t0, 3),
            "t_wall": round(time.time() - (time.monotonic() - t_start), 3),
            "dur_ms": round((time.monotonic() - t_start) * 1000),
            **extra,
        }
        _append_0600(self.log_path, json.dumps(rec) + "\n")


def _append_0600(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def _write_0600(path: Path, text: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, text.encode())
    finally:
        os.close(fd)


def _parse_combo(combo: str) -> tuple[str, bool, list[str]]:
    """'cmd+shift+a' -> ('a', False, ['cmd','shift']); 'return' -> ('36', True, [])."""
    parts = [p.strip() for p in combo.split("+")]
    if combo.endswith("++"):  # literal plus key, e.g. "shift++"
        parts = [*parts[:-2], "+"]
    *mod_parts, key = parts
    mods = []
    for m in mod_parts:
        if m.lower() not in _MODS:
            raise ValueError(f"unknown modifier {m!r}")
        mods.append(_MODS[m.lower()])
    if key.lower() in _NAMED_KEYS:
        return str(_NAMED_KEYS[key.lower()]), True, mods
    if len(key) != 1:
        raise ValueError(f"unknown key {key!r} (single character or one of {sorted(_NAMED_KEYS)})")
    return key, False, mods


def _observe_screen(state: BrokerState, app: str) -> list[Any]:
    t = time.monotonic()
    seq = state.next_seq()
    args = {"app": app}
    try:
        bounds = _osascript(_AS_FRONT_BOUNDS, *([app] if app else []))
        proc_name, _, rect = bounds.partition("|")
        data = _screenshot(state, rect, seq)
        obs_id = f"obs-{seq}-{secrets.token_hex(2)}"
        state.observations[obs_id] = seq
        state.log("observe_screen", args, seq, t, ok=True, obs_id=obs_id, img_bytes=len(data))
        meta = {
            "obs_id": obs_id,
            "seq": seq,
            "front_process": proc_name,
            "region": rect,
            "last_action_seq": state.last_action_seq,
        }
        return [
            TextContent(type="text", text=json.dumps(meta)),
            Image(data=data, format="jpeg"),
        ]
    except _ERRS as exc:
        state.log("observe_screen", args, seq, t, ok=False, err=type(exc).__name__)
        return [TextContent(type="text", text=f"error: {exc}")]


def _screenshot(state: BrokerState, rect: str, seq: int) -> bytes:
    """JPEG of *rect* (``x,y,w,h``; ``none`` = whole screen), scaled to ``max_px``."""
    fd, tmp = tempfile.mkstemp(suffix=".jpg", dir=state.call_dir)
    os.close(fd)
    try:
        cmd = ["screencapture", "-x", "-t", "jpg"]
        if rect != "none":
            cmd += ["-R", rect]
        subprocess.run([*cmd, tmp], check=True, timeout=10, capture_output=True)
        subprocess.run(
            ["sips", "-Z", str(state.capture.max_px), tmp],
            check=True,
            timeout=10,
            capture_output=True,
        )
        data = Path(tmp).read_bytes()
        if state.capture.keep_dir is not None:
            (state.capture.keep_dir / f"obs-{seq}.jpg").write_bytes(data)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return data


def _open_app(state: BrokerState, name: str) -> str:
    t = time.monotonic()
    seq = state.next_seq()
    proc = subprocess.run(
        ["open", "-a", name], capture_output=True, text=True, timeout=15, check=False
    )
    state.action_done(seq)
    ok = proc.returncode == 0
    state.log("open_app", {"name": name}, seq, t, ok=ok)
    return json.dumps({"seq": seq, "ok": ok, "error": proc.stderr.strip()[:200] if not ok else ""})


def _osa_action(
    state: BrokerState,
    tool: str,
    args: dict[str, str],
    script_args: Callable[[], tuple[str, ...]],
) -> str:
    """Run one System Events action; *script_args* returns ``(script, *argv)``.

    The action counts (``last_action_seq``) even when it fails, e.g. at argument parsing.
    """
    t = time.monotonic()
    seq = state.next_seq()
    try:
        script, *argv = script_args()
        _osascript(script, *argv)
        state.action_done(seq)
        state.log(tool, args, seq, t, ok=True)
        return json.dumps({"seq": seq, "ok": True})
    except _ERRS as exc:
        state.action_done(seq)
        state.log(tool, args, seq, t, ok=False, err=type(exc).__name__)
        return json.dumps({"seq": seq, "ok": False, "error": str(exc)[:300]})


def _key_script(combo: str, app: str) -> tuple[str, ...]:
    k, is_code, mods = _parse_combo(combo)
    return (_AS_KEY, app, k, "1" if is_code else "0", *mods)


def _list_ui(state: BrokerState, app: str) -> str:
    t = time.monotonic()
    seq = state.next_seq()
    try:
        out = _osascript(_AS_LIST_UI, app, "250", timeout=30)
        state.log("list_ui", {"app": app}, seq, t, ok=True, chars=len(out))
        if len(out) > _LIST_UI_MAX_CHARS:
            out = out[:_LIST_UI_MAX_CHARS] + "\n…(truncated)"
        return f"seq={seq}\n{out}"
    except _ERRS as exc:
        state.log("list_ui", {"app": app}, seq, t, ok=False, err=type(exc).__name__)
        return f"seq={seq} error: {exc}"


def _judge_finish(state: BrokerState, status: str, evidence_obs_id: str) -> tuple[str, str]:
    """``(recorded_status, reason)``: ``done`` needs an observation newer than the last action."""
    if status not in {"done", "failed", "unknown_partial", "needs_confirmation"}:
        return "failed", f"invalid status {status!r}"
    if status == "done":
        obs_seq = state.observations.get(evidence_obs_id)
        if obs_seq is None:
            return "unknown_partial", "evidence_obs_id is not a known observation"
        if obs_seq < state.last_action_seq:
            return "unknown_partial", (
                f"evidence observation seq {obs_seq} is older than the last action "
                f"seq {state.last_action_seq}"
            )
    return status, ""


def _finish(state: BrokerState, status: str, summary: str, evidence_obs_id: str) -> str:
    t = time.monotonic()
    seq = state.next_seq()
    args = {"status": status, "summary": summary, "evidence_obs_id": evidence_obs_id}
    recorded, reason = _judge_finish(state, status, evidence_obs_id)
    rec = {
        "seq": seq,
        "requested_status": status,
        "recorded_status": recorded,
        "reason": reason,
        "evidence_obs_id": evidence_obs_id or None,
        "evidence_seq": state.observations.get(evidence_obs_id),
        "last_action_seq": state.last_action_seq,
        "summary_len": len(summary),
        "t_s": round(t - state.t0, 3),
    }
    state.finishes.append(rec)
    _write_0600(state.finish_path, json.dumps({"finishes": state.finishes}, indent=2) + "\n")
    state.log("finish", args, seq, t, ok=True, recorded_status=recorded)
    msg = {"recorded_status": recorded, "reason": reason}
    if recorded != status:
        msg["hint"] = "observe_screen again, then call finish again with the new obs_id"
    return json.dumps(msg)


def build_server(state: BrokerState) -> MCPServer:
    server = MCPServer("mac-broker-proto")

    @server.tool(structured_output=False)
    def observe_screen(app: str = "") -> list[Any]:
        """Screenshot of the front window (or of `app`'s first window) as an image.

        Returns the image plus an obs_id and seq. Use the obs_id as evidence in finish().
        """
        return _observe_screen(state, app)

    @server.tool(structured_output=False)
    def open_app(name: str) -> str:
        """Open (launch or bring to front) a macOS application by name, e.g. 'Calculator'."""
        return _open_app(state, name)

    @server.tool(structured_output=False)
    def key(combo: str, app: str = "") -> str:
        """Press a key or combo in `app` (activated first), e.g. '2', 'cmd+c', 'return', 'escape'.

        Combos join modifiers (cmd, shift, alt, ctrl) and one key with '+'.
        """
        return _osa_action(
            state, "key", {"combo": combo, "app": app}, lambda: _key_script(combo, app)
        )

    @server.tool(structured_output=False)
    def type_text(app: str, text: str) -> str:
        """Type literal text into `app` (activated first) via System Events keystrokes."""
        return _osa_action(
            state, "type_text", {"app": app, "text": text}, lambda: (_AS_TYPE, app, text)
        )

    @server.tool(structured_output=False)
    def list_ui(app: str) -> str:
        """Text dump of `app`'s first window UI elements: 'role | name | description | value'.

        Truncated. Cheaper than a screenshot; NOT valid as finish() evidence.
        """
        return _list_ui(state, app)

    @server.tool(structured_output=False)
    def finish(status: str, summary: str, evidence_obs_id: str = "") -> str:
        """Report the final result. status: done | failed | unknown_partial | needs_confirmation.

        'done' requires evidence_obs_id from an observe_screen taken AFTER your last action;
        otherwise the broker records unknown_partial. Then return the recorded status as
        your structured result.
        """
        return _finish(state, status, summary, evidence_obs_id)

    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Prototype stdio MCP Mac broker for spike S-OPERATOR.",
        epilog="Example: broker_proto.py -c ~/.local/state/mac-voice/operator/<call-id> -m 800",
    )
    ap.add_argument("-c", "--call-dir", required=True, type=Path, help="0700 per-call directory")
    ap.add_argument(
        "-m", "--max-px", type=int, default=800, help="screenshot max edge in px (default 800)"
    )
    ap.add_argument(
        "-k",
        "--keep-dir",
        type=Path,
        default=None,
        help="debug only: also save each screenshot here (never in fixtures)",
    )
    args = ap.parse_args(argv)
    call_dir = args.call_dir.expanduser().resolve()
    if not call_dir.is_dir():
        print(f"call dir {call_dir} does not exist", file=sys.stderr)
        return 2
    state = BrokerState(call_dir, args.max_px, args.keep_dir)
    build_server(state).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
