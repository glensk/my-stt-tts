"""Mac broker: the local stdio MCP server through which the Mac operator acts.

The operator (``mac_operator.py``) runs ``claude -p`` with this broker as its only tool
source. Every primitive goes through :meth:`Broker.call`, which classifies it against the
risk table (PLAN_claude-bridge.md 7.4, :mod:`my_stt_tts.mac_risk`) before anything
touches the Mac:

* SAFE runs at once: opening URLs and apps, observing / reading (screenshots, UI dumps,
  CSS-selector reads of the Safari page through a fixed read-only script), typing into
  ordinary apps, navigation keys, Return in a search field / address bar, Focus /
  Do-Not-Disturb, volume, brightness, media.
* CONFIRM blocks: closing tabs / windows / apps, sending / posting / mail / messages,
  deleting / moving / renaming / saving files, purchases, security / privacy / network
  settings, installs, typing into shells, Return outside a search field / address bar,
  URLs with a long query, every ``applescript``, anything unrecognised. The broker
  writes ``<call-dir>/blocked-<sha>.json`` (tool, normalised-args sha256, a one-line
  summary, a nonce; 0600) and waits up to 60 s for ``allow-<sha>.json`` carrying the same
  nonce, which the daemon writes after a spoken ``confirm_action``. Exactly that call
  then runs once (the allow file is consumed); a timeout returns ``denied: not
  confirmed``.
* REFUSE never runs: ``applescript`` containing ``do shell script`` (or the ways around
  it: ``run script``, ``load script``, the raw ``sysoexec`` event), non-http(s) URLs, app
  names that are paths.

``finish(done)`` needs an observation taken AFTER the last action, else it is recorded as
``unknown_partial``; ``<call-dir>/finish.json`` is written the moment ``finish`` is called
so the operator can return without waiting for the structured-result turn.
``<call-dir>/evidence.json`` (0600) is rewritten on every action and observation —
``{"last_action_seq": n, "observations": {obs_id: seq}}`` (the last
``EVIDENCE_KEEP``) — so the operator can verify a structured ``done`` when the model
never called ``finish``. A failed rewrite deletes the file (fail closed).

Audit: ``<call-dir>/broker_audit.jsonl`` (0600) — tool, args sha256, risk, outcome,
duration; never content. Screenshots live only as temp files in the call dir and are
deleted right after they are read.

The module imports only the standard library at import time (``mcp`` loads in
:func:`build_server`), so the server starts fast. Run by file path with ``python -I``.

Usage:
    mac_broker.py -c CALL_DIR [-r ROOT] [-m MAX_PX] [-t CONFIRM_TIMEOUT]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run by file path (python -I mac_broker.py): find the package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from my_stt_tts.mac_risk import (  # noqa: E402  # pylint: disable=wrong-import-position
    CONTROL_RE,
    NAMED_KEYS,
    REFUSE,
    SAFE,
    RiskContext,
    allow_path,
    append_0600,
    args_sha256,
    blocked_path,
    classify,
    focused_ax,
    hit_test_ax,
    parse_combo,
    shell_script_refusal,
    validate_app_name,
    validate_url,
    write_0600,
)

OPERATOR_ROOT = Path.home() / ".local" / "state" / "mac-voice" / "operator"
CONFIRM_TIMEOUT_S = 60.0
CONFIRM_POLL_S = 0.2
DEFAULT_MAX_PX = 800
RESULT_CAP = 6000
DOM_CAP = 8192  # total characters a safari_dom reply may carry
DOM_LIMIT_MAX = 50
DOM_SELECTOR_CAP = 500
TEXT_CAP = 2000
FINISH_STATUSES = ("done", "failed", "unknown_partial", "needs_confirmation")
FINISH_SUMMARY_CAP = 300
EVIDENCE_KEEP = 50  # observations kept in evidence.json

DENIED = "denied: not confirmed"

OSASCRIPT = "/usr/bin/osascript"
OPEN = "/usr/bin/open"
SCREENCAPTURE = "/usr/sbin/screencapture"
SIPS = "/usr/bin/sips"
SHORTCUTS = "/usr/bin/shortcuts"

Runner = Callable[[list[str], float], subprocess.CompletedProcess[str]]


def default_runner(argv: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` (never a shell), capture text output, never raise on the exit code."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)


# -- side effects (all fakeable) -------------------------------------------------------------
_AS_FRONT_APP = (
    'tell application "System Events" to return name of first process whose frontmost is true'
)
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
_AS_DESKTOP = 'tell application "Finder" to get bounds of window of desktop'
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
    if appName is not "" then tell application appName to activate
    delay 0.15
    tell application "System Events" to keystroke (item 2 of argv)
end run
"""
_AS_ACTIVATE = "on run argv\ntell application (item 1 of argv) to activate\nend run"
_AS_LIST_UI = """
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
    set AppleScript's text item delimiters to linefeed
    return out as text
end run
"""
_AS_CLICK_ELEMENT = """
on run argv
    set appName to item 1 of argv
    set target to item 2 of argv
    tell application appName to activate
    delay 0.15
    tell application "System Events"
        tell process appName
            set els to entire contents of window 1
        end tell
        repeat with e in els
            set nm to ""
            set ds to ""
            try
                set nm to name of e as text
            end try
            try
                set ds to description of e as text
            end try
            if nm is target or ds is target then
                click e
                return role of e
            end if
        end repeat
    end tell
    error "no element named " & target
end run
"""
_AS_MENU = """
on run argv
    set appName to item 1 of argv
    tell application appName to activate
    delay 0.15
    tell application "System Events"
        tell process appName
            set n to count of argv
            if n = 3 then
                click menu item (item 3 of argv) of menu (item 2 of argv) of ¬
                    menu bar item (item 2 of argv) of menu bar 1
            else if n = 4 then
                click menu item (item 4 of argv) of menu (item 3 of argv) of ¬
                    menu item (item 3 of argv) of menu (item 2 of argv) of ¬
                    menu bar item (item 2 of argv) of menu bar 1
            else
                error "menu path needs 2 or 3 levels"
            end if
        end tell
    end tell
end run
"""
# safari_dom's only script: the selector goes in as a JSON string literal, never as code.
# Per match: tag, text (<= 200 chars), href, aria-label, role, value of form fields except
# password / hidden inputs (their type is reported so the broker can re-check).
_DOM_JS = (
    "(function(){try{var sel=__SEL__,lim=__LIM__,out=[],els=document.querySelectorAll(sel);"
    "function cut(v,n){return String(v).trim().slice(0,n);}"
    "for(var i=0;i<els.length&&out.length<lim;i++){var e=els[i],o={tag:e.tagName.toLowerCase()};"
    "var t=cut(e.innerText||e.textContent||'',200);if(t)o.text=t;"
    "if(e.hasAttribute('href'))o.href=cut(e.href||e.getAttribute('href'),500);"
    "if(e.hasAttribute('aria-label'))o['aria-label']=cut(e.getAttribute('aria-label'),200);"
    "if(e.hasAttribute('role'))o.role=cut(e.getAttribute('role'),50);"
    "var tg=o.tag,ty=String(e.type||'').toLowerCase(),ac=String(e.autocomplete||'');"
    "if(tg==='input')o.type=ty;"
    "if((tg==='input'||tg==='textarea'||tg==='select')&&ty!=='password'&&ty!=='hidden'"
    "&&!/password/i.test(ac)&&typeof e.value==='string')o.value=cut(e.value,200);"
    "out.push(o);}"
    "return JSON.stringify({count:els.length,items:out});"
    "}catch(err){return 'error: '+(err&&err.message?err.message:String(err));}})()"
)
_DOM_SECRET_TYPES = frozenset({"password", "hidden"})


def safari_dom_js(selector: str, limit: int) -> str:
    """The fixed read-only script for ``safari_dom``; ``selector`` is inserted only as a
    JSON string literal (ASCII-escaped), ``limit`` as a clamped integer."""
    if not isinstance(selector, str) or not selector.strip():
        raise ValueError("safari_dom needs a CSS selector")
    if len(selector) > DOM_SELECTOR_CAP:
        raise ValueError(f"selector longer than {DOM_SELECTOR_CAP} characters")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ValueError("limit must be an integer")
    lim = max(1, min(limit, DOM_LIMIT_MAX))
    return _DOM_JS.replace("__LIM__", str(lim)).replace(
        "__SEL__", json.dumps(selector, ensure_ascii=True)
    )


def dom_reply(raw: str) -> str:
    """The page's answer, re-checked: no values of password / hidden inputs, <= DOM_CAP."""
    try:
        data = json.loads(raw)
    except ValueError:
        return raw[:DOM_CAP]  # the script's own "error: …" string
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        return "error: unexpected reply from the page"
    items: list[dict[str, Any]] = []
    for item in data["items"]:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).lower() in _DOM_SECRET_TYPES:
            item.pop("value", None)
        items.append(item)
    out = {"count": data.get("count"), "truncated": False, "items": items}
    while len(text := json.dumps(out, ensure_ascii=False)) > DOM_CAP and items:
        items.pop()
        out["truncated"] = True
    return text[:DOM_CAP]


_AS_SAFARI_JS = """
on run argv
    tell application "Safari"
        if (count of windows) = 0 then error "no Safari window"
        set wid to id of front window
        return do JavaScript (item 1 of argv) in current tab of window id wid
    end tell
end run
"""


def _quartz_click(x: float, y: float) -> None:
    """A real left click at screen point (x, y) via Quartz events (pyobjc, lazy import)."""
    import Quartz  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error

    # pylint: disable=no-member  # pyobjc members are generated at runtime
    point = (float(x), float(y))
    for kind in (
        Quartz.kCGEventMouseMoved,
        Quartz.kCGEventLeftMouseDown,
        Quartz.kCGEventLeftMouseUp,
    ):
        event = Quartz.CGEventCreateMouseEvent(None, kind, point, Quartz.kCGMouseButtonLeft)
        Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        time.sleep(0.05)


def jpeg_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's SOF marker, or None."""
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        length = int.from_bytes(data[i + 2 : i + 4], "big")
        if marker in {0xC0, 0xC1, 0xC2}:
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            return width, height
        i += 2 + length
    return None


# -- the broker ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Shot:
    """One observation: metadata for the model + the JPEG bytes (never written to disk)."""

    meta: dict[str, Any]
    data: bytes = field(repr=False)


@dataclass(frozen=True)
class Observation:
    seq: int
    region: tuple[float, float, float, float]  # screen points: x, y, w, h
    size: tuple[int, int]  # image pixels: w, h


Reply = list[str | Shot]

_ACTION_TOOLS = frozenset(
    {"click", "type_text", "key", "open_url", "open_app", "run_shortcut", "menu_select",
     "applescript"}
)  # fmt: skip
_ERRS = (RuntimeError, ValueError, OSError, subprocess.SubprocessError)


class Broker:  # pylint: disable=too-many-instance-attributes  # the broker's whole state
    """Risk-checked Mac primitives for one operator call (one call dir)."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        call_dir: Path,
        *,
        runner: Runner = default_runner,
        max_px: int = DEFAULT_MAX_PX,
        confirm_timeout: float = CONFIRM_TIMEOUT_S,
        poll: float = CONFIRM_POLL_S,
        clicker: Callable[[float, float], None] = _quartz_click,
        hit_test: Callable[[float, float], Mapping[str, str] | None] | None = hit_test_ax,
        focused: Callable[[], Mapping[str, str] | None] | None = focused_ax,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.call_dir = call_dir
        self.runner = runner
        self.max_px = max_px
        self.confirm_timeout = confirm_timeout
        self.poll = poll
        self.clicker = clicker
        self.clock = clock
        self.sleep = sleep
        self.ctx = RiskContext(self.front_app, hit_test, focused)
        self.seq = 0
        self.last_action_seq = 0
        self.observations: dict[str, Observation] = {}
        self.finished: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self.t0 = clock()

    # -- bookkeeping ------------------------------------------------------------------
    @property
    def audit_path(self) -> Path:
        return self.call_dir / "broker_audit.jsonl"

    @property
    def finish_path(self) -> Path:
        return self.call_dir / "finish.json"

    @property
    def evidence_path(self) -> Path:
        return self.call_dir / "evidence.json"

    def _next_seq(self) -> int:
        with self._lock:
            self.seq += 1
            return self.seq

    def _acted(self, seq: int) -> None:
        with self._lock:
            self.last_action_seq = max(self.last_action_seq, seq)
            self._write_evidence()

    def _write_evidence(self) -> None:
        """Rewrite ``evidence.json``; call with ``_lock`` held so writes stay ordered."""
        recent = sorted(self.observations.items(), key=lambda kv: kv[1].seq)[-EVIDENCE_KEEP:]
        record = {
            "last_action_seq": self.last_action_seq,
            "observations": {obs_id: obs.seq for obs_id, obs in recent},
        }
        try:
            write_0600(self.evidence_path, json.dumps(record) + "\n")
        except OSError:  # a stale record could vouch for an old observation: drop it
            with contextlib.suppress(OSError):
                self.evidence_path.unlink(missing_ok=True)

    def _audit(self, record: Mapping[str, Any]) -> None:
        append_0600(self.audit_path, json.dumps(dict(record), sort_keys=True) + "\n")

    # -- helpers ----------------------------------------------------------------------
    def osascript(self, script: str, *args: str, timeout: float = 15.0) -> str:
        """``osascript -e <script> -- <args…>`` — values only via argv, never spliced."""
        proc = self.runner([OSASCRIPT, "-e", script, "--", *args], timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"osascript failed: {(proc.stderr or '').strip()[:300]}")
        return (proc.stdout or "").strip()

    def front_app(self) -> str:
        return self.osascript(_AS_FRONT_APP, timeout=5.0)

    # -- the single entry point -------------------------------------------------------
    def call(self, tool: str, args: Mapping[str, Any]) -> Reply:
        """Classify, block for confirmation if needed, run, audit. Never raises."""
        args = dict(args)
        start = self.clock()
        sha = args_sha256(tool, args)
        if tool == "click":
            try:
                args["_point"] = self._click_point(args)
            except ValueError as exc:
                self._audit_call(tool, sha, "safe", "error", start=start)
                return [f"error: {exc}"]
        verdict = classify(tool, args, self.ctx)
        if verdict.risk == REFUSE:
            self._audit_call(tool, sha, verdict.risk, "refused", start=start)
            return [f"refused: {verdict.reason}"]
        if verdict.risk != SAFE and not self._confirmed(tool, sha, verdict.summary):
            self._audit_call(tool, sha, verdict.risk, "denied", start=start)
            return [DENIED]
        try:
            reply = self._dispatch(tool, args)
            outcome = "ok"
        except _ERRS as exc:
            reply, outcome = [f"error: {str(exc)[:300]}"], "error"
        if tool in _ACTION_TOOLS and args.get("observe") and outcome == "ok":
            self.sleep(0.3)
            reply = [*reply, *self.observe_screen(str(args.get("app", "")))]
        self._audit_call(tool, sha, verdict.risk, outcome, start=start)
        return reply

    def _audit_call(self, tool: str, sha: str, risk: str, outcome: str, *, start: float) -> None:
        self._audit(
            {
                "tool": tool,
                "args_sha256": sha,
                "risk": risk,
                "outcome": outcome,
                "t_s": round(start - self.t0, 3),
                "dur_ms": round((self.clock() - start) * 1000),
            }
        )

    def _confirmed(self, tool: str, sha: str, summary: str) -> bool:
        """Write the blocked file, wait for the matching allow file, consume it."""
        blocked, allow = blocked_path(self.call_dir, sha), allow_path(self.call_dir, sha)
        nonce = secrets.token_hex(16)
        allow.unlink(missing_ok=True)  # a stale allow never covers a new block
        record = {"tool": tool, "sha256": sha, "summary": summary, "nonce": nonce}
        write_0600(blocked, json.dumps(record) + "\n")
        deadline = self.clock() + self.confirm_timeout
        try:
            while self.clock() < deadline:
                if self._take_allow(allow, sha, nonce):
                    return True
                self.sleep(self.poll)
            return self._take_allow(allow, sha, nonce)
        finally:
            with contextlib.suppress(OSError):
                current = json.loads(blocked.read_text())
                if current.get("nonce") == nonce:
                    blocked.unlink()

    @staticmethod
    def _take_allow(allow: Path, sha: str, nonce: str) -> bool:
        try:
            data = json.loads(allow.read_text())
        except (OSError, ValueError):
            return False
        try:
            allow.unlink()  # consumed: one allow → exactly one run
        except FileNotFoundError:
            return False
        return data.get("sha256") == sha and secrets.compare_digest(
            str(data.get("nonce", "")), nonce
        )

    def _dispatch(self, tool: str, args: Mapping[str, Any]) -> Reply:
        handlers: dict[str, Callable[[Mapping[str, Any]], Reply]] = {
            "observe_screen": lambda a: self.observe_screen(
                str(a.get("app", "")), bool(a.get("full_screen"))
            ),
            "list_ui": lambda a: [self._list_ui(str(a.get("app", "")))],
            "safari_dom": lambda a: [self._safari_dom(a.get("selector", ""), a.get("limit", 20))],
            "click": self._click,
            "type_text": lambda a: [self._type_text(str(a.get("app", "")), str(a["text"]))],
            "key": lambda a: [self._key(str(a.get("combo", "")), str(a.get("app", "")))],
            "open_url": lambda a: [self._open_url(str(a.get("url", "")))],
            "open_app": lambda a: [self._open_app(str(a.get("name", "")))],
            "run_shortcut": lambda a: [self._run_shortcut(str(a.get("name", "")))],
            "menu_select": lambda a: [self._menu(str(a.get("app", "")), str(a.get("path", "")))],
            "applescript": lambda a: [self._applescript(str(a.get("script", "")))],
            "finish": lambda a: [
                self._finish(
                    str(a.get("status", "")),
                    str(a.get("summary", "")),
                    str(a.get("evidence_obs_id") or ""),
                )
            ],
        }
        handler = handlers.get(tool)
        if handler is None:
            raise ValueError(f"unknown tool {tool!r}")
        return handler(args)

    def _action(self, run: Callable[[], str]) -> str:
        """Run one mutating primitive; it counts as an action even when it fails."""
        seq = self._next_seq()
        try:
            detail = run()
        finally:
            self._acted(seq)
        return json.dumps({"seq": seq, "ok": True, **({"result": detail} if detail else {})})

    # -- observing --------------------------------------------------------------------
    def observe_screen(self, app: str = "", full_screen: bool = False) -> Reply:
        """Screenshot of the front window / ``app``'s window / the screen + its obs_id."""
        seq = self._next_seq()
        proc_name, region = self._region(app, full_screen)
        data = self._capture(region)
        size = jpeg_size(data) or (self.max_px, self.max_px)
        obs_id = f"obs-{seq}-{secrets.token_hex(2)}"
        with self._lock:
            self.observations[obs_id] = Observation(seq, region, size)
            last_action = self.last_action_seq
            self._write_evidence()
        meta = {
            "obs_id": obs_id,
            "seq": seq,
            "front_process": proc_name,
            "image_px": list(size),
            "last_action_seq": last_action,
            "hint": "click(x, y, obs_id) takes pixel coordinates in this image",
        }
        return [json.dumps(meta), Shot(meta, data)]

    def _region(self, app: str, full_screen: bool) -> tuple[str, tuple[float, float, float, float]]:
        name = ""
        if not full_screen:
            out = self.osascript(_AS_FRONT_BOUNDS, *([app] if app else []))
            name, _, rect = out.partition("|")
            if rect and rect != "none":
                x, y, w, h = (float(v) for v in rect.split(","))
                return name, (x, y, w, h)
        bounds = [float(v) for v in self.osascript(_AS_DESKTOP).replace(" ", "").split(",")]
        x0, y0, x1, y1 = bounds
        return name, (x0, y0, x1 - x0, y1 - y0)

    def _capture(self, region: tuple[float, float, float, float]) -> bytes:
        """JPEG of ``region`` scaled to ``max_px``; the temp file is deleted at once."""
        fd, tmp = _mkstemp(self.call_dir, ".jpg")
        os.close(fd)
        try:
            rect = ",".join(str(round(v)) for v in region)
            for argv in (
                [SCREENCAPTURE, "-x", "-t", "jpg", "-R", rect, tmp],
                [SIPS, "-Z", str(self.max_px), tmp],
            ):
                proc = self.runner(argv, 10.0)
                if proc.returncode != 0:
                    raise RuntimeError(f"{Path(argv[0]).name} failed")
            data = Path(tmp).read_bytes()
            if jpeg_size(data) is None:  # screencapture can fail with exit status 0
                raise RuntimeError("screenshot failed (Screen Recording permission?)")
            return data
        finally:
            Path(tmp).unlink(missing_ok=True)

    def _list_ui(self, app: str) -> str:
        seq = self._next_seq()
        out = self.osascript(_AS_LIST_UI, app, "250", timeout=30.0)
        if len(out) > RESULT_CAP:
            out = out[:RESULT_CAP] + "\n…(truncated)"
        return f"seq={seq} (not valid as finish evidence)\n{out}"

    def _safari_dom(self, selector: Any, limit: Any) -> str:
        js = safari_dom_js(selector, limit)
        seq = self._next_seq()
        out = self.osascript(_AS_SAFARI_JS, js, timeout=15.0)
        return f"seq={seq}\n{dom_reply(out)}"

    # -- acting -----------------------------------------------------------------------
    def _click_point(self, args: Mapping[str, Any]) -> tuple[float, float] | None:
        if str(args.get("element", "") or "").strip():
            return None
        x, y = args.get("x"), args.get("y")
        if x is None or y is None:
            raise ValueError("click needs x and y, or element (with app)")
        obs_id = str(args.get("obs_id", "") or "")
        if not obs_id:
            return float(x), float(y)
        with self._lock:
            obs = self.observations.get(obs_id)
        if obs is None:
            raise ValueError(f"unknown obs_id {obs_id!r}")
        rx, ry, rw, rh = obs.region
        iw, ih = obs.size
        if not (0 <= float(x) <= iw and 0 <= float(y) <= ih):
            raise ValueError("x/y outside the observation image")
        return rx + float(x) * rw / iw, ry + float(y) * rh / ih

    def _click(self, args: Mapping[str, Any]) -> Reply:
        app = str(args.get("app", "") or "")
        element = str(args.get("element", "") or "").strip()
        point = args.get("_point")

        def run() -> str:
            if element:
                if not app:
                    raise ValueError("click by element needs app")
                return self.osascript(_AS_CLICK_ELEMENT, app, element, timeout=30.0)
            assert isinstance(point, tuple)
            if app:
                self.osascript(_AS_ACTIVATE, app)
                self.sleep(0.15)
            self.clicker(point[0], point[1])
            return ""

        return [self._action(run)]

    def _type_text(self, app: str, text: str) -> str:
        if len(text) > TEXT_CAP:
            raise ValueError(f"text longer than {TEXT_CAP} characters")
        return self._action(lambda: self.osascript(_AS_TYPE, app, text))

    def _key(self, combo: str, app: str) -> str:
        def run() -> str:
            key, mods = parse_combo(combo)
            code = NAMED_KEYS.get(key)
            k, is_code = (str(code), "1") if code is not None else (key, "0")
            return self.osascript(_AS_KEY, app, k, is_code, *mods)

        return self._action(run)

    def _open_url(self, url: str) -> str:
        target = validate_url(url)
        return self._action(lambda: self._run_checked([OPEN, "-a", "Safari", target]))

    def _open_app(self, name: str) -> str:
        app = validate_app_name(name)
        return self._action(lambda: self._run_checked([OPEN, "-a", app]))

    def _run_shortcut(self, name: str) -> str:
        return self._action(lambda: self._run_checked([SHORTCUTS, "run", name], timeout=60.0))

    def _menu(self, app: str, path: str) -> str:
        parts = [p.strip() for p in re.split(r">|→|/", path) if p.strip()]

        def run() -> str:
            if not app:
                raise ValueError("menu_select needs app")
            if len(parts) not in (2, 3):
                raise ValueError("menu path needs 2 or 3 levels, e.g. 'File > Close Window'")
            return self.osascript(_AS_MENU, app, *parts)

        return self._action(run)

    def _applescript(self, script: str) -> str:
        if shell_script_refusal(script):  # classify() already refused; belt and braces
            raise ValueError("refused")
        return self._action(lambda: self.osascript(script, timeout=30.0)[:RESULT_CAP])

    def _run_checked(self, argv: list[str], timeout: float = 15.0) -> str:
        proc = self.runner(argv, timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"{Path(argv[0]).name} failed: {(proc.stderr or '').strip()[:200]}")
        return ""

    # -- finishing --------------------------------------------------------------------
    def judge_finish(self, status: str, evidence_obs_id: str) -> tuple[str, str]:
        """``(recorded_status, reason)``: ``done`` needs an observation after the last action."""
        if status not in FINISH_STATUSES:
            return "failed", f"invalid status {status!r}"
        if status != "done":
            return status, ""
        with self._lock:
            obs = self.observations.get(evidence_obs_id)
            last = self.last_action_seq
        if obs is None:
            return "unknown_partial", "evidence_obs_id is not a known observation"
        if obs.seq < last:
            return "unknown_partial", "the evidence observation is older than the last action"
        return "done", ""

    def _finish(self, status: str, summary: str, evidence_obs_id: str) -> str:
        seq = self._next_seq()
        recorded, reason = self.judge_finish(status, evidence_obs_id)
        record = {
            "seq": seq,
            "status": recorded,
            "requested_status": status,
            "reason": reason,
            "summary": " ".join(CONTROL_RE.sub(" ", summary).split())[:FINISH_SUMMARY_CAP],
            "evidence_obs_id": evidence_obs_id or None,
            "last_action_seq": self.last_action_seq,
        }
        with self._lock:
            self.finished = record
        write_0600(self.finish_path, json.dumps(record) + "\n")
        return json.dumps({"recorded_status": recorded, "reason": reason, "final": True})


def _mkstemp(directory: Path, suffix: str) -> tuple[int, str]:
    # not a dot-file: screencapture refuses hidden destinations (and still exits 0)
    path = directory / f"shot-{secrets.token_hex(6)}{suffix}"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return fd, str(path)


# -- the MCP server --------------------------------------------------------------------------
def _to_content(reply: Reply) -> list[Any]:
    from mcp.server.mcpserver.utilities.types import (  # pylint: disable=import-outside-toplevel
        Image,
    )
    from mcp.types import TextContent  # pylint: disable=import-outside-toplevel

    out: list[Any] = []
    for part in reply:
        if isinstance(part, Shot):
            out.append(Image(data=part.data, format="jpeg"))
        else:
            out.append(TextContent(type="text", text=part))
    return out


def build_server(broker: Broker) -> Any:  # noqa: C901  # one small closure per primitive
    """The stdio MCP server (``mcp`` imported here, not at module import)."""
    from mcp.server.mcpserver import MCPServer  # pylint: disable=import-outside-toplevel

    server = MCPServer("mac")

    def run(tool: str, **args: Any) -> list[Any]:
        return _to_content(broker.call(tool, args))

    @server.tool(structured_output=False)
    def observe_screen(app: str = "", full_screen: bool = False) -> list[Any]:
        """Screenshot of the front window (or `app`'s first window, or the whole screen).

        Returns the image plus obs_id / seq. Use the obs_id as finish() evidence and with
        click(x, y, obs_id) to click at pixel coordinates of this image.
        """
        return run("observe_screen", app=app, full_screen=full_screen)

    @server.tool(structured_output=False)
    def list_ui(app: str) -> list[Any]:
        """Text dump of `app`'s first window: 'role | name | description | value' (truncated).

        Cheaper than a screenshot; not valid as finish() evidence.
        """
        return run("list_ui", app=app)

    @server.tool(structured_output=False)
    def safari_dom(selector: str, limit: int = 20) -> list[Any]:
        """Read elements of the front Safari tab matching a CSS selector (read-only).

        Returns JSON {count, truncated, items}: per match (at most `limit`, max 50) its tag,
        text (first 200 chars), href, aria-label, role and form-field value (never for
        password fields). Example: safari_dom('h3 a', 10). No JavaScript can be run.
        """
        return run("safari_dom", selector=selector, limit=limit)

    @server.tool(structured_output=False)
    def click(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        x: float | None = None,
        y: float | None = None,
        obs_id: str = "",
        element: str = "",
        app: str = "",
        observe: bool = False,
    ) -> list[Any]:
        """Left-click. Either x/y (pixels of observation `obs_id`; screen points without it)
        or `element` = the exact name/description of a UI element of `app` (see list_ui).

        observe=true returns a fresh screenshot after the click (saves a turn).
        """
        return run("click", x=x, y=y, obs_id=obs_id, element=element, app=app, observe=observe)

    @server.tool(structured_output=False)
    def type_text(app: str, text: str, observe: bool = False) -> list[Any]:
        """Type literal text into `app` (activated first). Use key('return') to submit."""
        return run("type_text", app=app, text=text, observe=observe)

    @server.tool(structured_output=False)
    def key(combo: str, app: str = "", observe: bool = False) -> list[Any]:
        """Press a key or combo in `app` (activated first): '2', 'cmd+l', 'return', 'escape'.

        Modifiers cmd, shift, alt, ctrl joined with '+'; named keys: return, enter, tab, space,
        delete, escape, left, right, up, down, home, end, pageup, pagedown.
        """
        return run("key", combo=combo, app=app, observe=observe)

    @server.tool(structured_output=False)
    def open_url(url: str, observe: bool = False) -> list[Any]:
        """Open an http(s) URL in Safari (bare hosts get https; 'youtube', 'jellyfin')."""
        return run("open_url", url=url, observe=observe)

    @server.tool(structured_output=False)
    def open_app(name: str, observe: bool = False) -> list[Any]:
        """Open (launch or bring to front) an installed app by name, e.g. 'Calculator'."""
        return run("open_app", name=name, observe=observe)

    @server.tool(structured_output=False)
    def run_shortcut(name: str, observe: bool = False) -> list[Any]:
        """Run one of the user's Shortcuts by its exact name."""
        return run("run_shortcut", name=name, observe=observe)

    @server.tool(structured_output=False)
    def menu_select(app: str, path: str, observe: bool = False) -> list[Any]:
        """Choose a menu item of `app`: 'View > Enter Full Screen' (2 or 3 levels)."""
        return run("menu_select", app=app, path=path, observe=observe)

    @server.tool(structured_output=False)
    def applescript(script: str) -> list[Any]:
        """Run AppleScript source (always held for the user's spoken confirmation).

        Shell commands (do shell script, run script) are refused.
        """
        return run("applescript", script=script)

    @server.tool(structured_output=False)
    def finish(status: str, summary: str, evidence_obs_id: str = "") -> list[Any]:
        """Report the final result ONCE: status done | failed | unknown_partial |
        needs_confirmation, a one-sentence speakable summary, and for 'done' the obs_id of
        an observe_screen taken AFTER your last action (else it is recorded unknown_partial).
        """
        return run("finish", status=status, summary=summary, evidence_obs_id=evidence_obs_id)

    return server


def check_call_dir(call_dir: Path, root: Path) -> Path:
    """The resolved call dir, or ValueError when it is outside ``root`` or not private."""
    resolved = call_dir.expanduser().resolve()
    base = root.expanduser().resolve()
    if not resolved.is_relative_to(base) or resolved == base:
        raise ValueError(f"call dir {resolved} is outside {base}")
    if not resolved.is_dir():
        raise ValueError(f"call dir {resolved} does not exist")
    if resolved.stat().st_mode & 0o077:
        raise ValueError(f"call dir {resolved} is readable by others (needs 0700)")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Risk-checked Mac broker (stdio MCP server) for the voice operator.",
        epilog="Example: mac_broker.py -c ~/.local/state/mac-voice/operator/<call-id> -m 800",
    )
    parser.add_argument("-c", "--call-dir", required=True, type=Path, help="0700 per-call dir")
    parser.add_argument(
        "-r", "--root", type=Path, default=OPERATOR_ROOT, help="allowed parent of call dirs"
    )
    parser.add_argument(
        "-m", "--max-px", type=int, default=DEFAULT_MAX_PX, help="screenshot max edge (px)"
    )
    parser.add_argument(
        "-t",
        "--confirm-timeout",
        type=float,
        default=CONFIRM_TIMEOUT_S,
        help="seconds a blocked call waits for confirmation (default 60)",
    )
    args = parser.parse_args(argv)
    try:
        call_dir = check_call_dir(args.call_dir, args.root)
    except ValueError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2
    broker = Broker(call_dir, max_px=args.max_px, confirm_timeout=args.confirm_timeout)
    build_server(broker).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
