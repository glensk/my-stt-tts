"""Mac fast path for the voice bridge: websites, apps, volume, brightness, Safari media.

Client tools (PLAN_claude-bridge 7.3), registered on the
:class:`~my_stt_tts.bridge.BridgeController` by :func:`register` when
``MAC_VOICE_MAC_CONTROL=1``:

=========================  =====================================================================
``open_url(target)``       any http(s) site in Safari; shortcuts ``youtube`` / ``jellyfin``
``open_app(name)``         any app in /Applications, /System/Applications, ~/Applications
``set_volume(…)``          ``level`` 0–100, ``step`` up / down / ±N, or ``mute`` true / false
``set_brightness(step)``   built-in display only: up / down / ±N key presses (1/16 each)
``media(command, s?)``     play | pause | seek ±s | next | previous | fullscreen (YouTube, Jellyfin)
``youtube_play_first()``   open + play the first visible video on a YouTube page
=========================  =====================================================================

Every tool: a capability from the latest authorised transcript, arguments derivable from
it, then :func:`~my_stt_tts.bridge.before_mutation` (call's cancellation token, 3 s
monotonic deadline, capability consumed) under the controller's mutation lock, right
before acting. Subprocesses are argv lists with timeouts through an injectable runner.
Successes return a short neutral ``ok: …`` (the agent stays silent), refusals
``refused: …`` and failures ``failed: …`` (also reported as a
:class:`~my_stt_tts.bridge.Problem`). One log line per call, never content.

``mac-voice -D`` runs :func:`doctor_main`; while a doctor check fails :func:`register`
registers only stubs that answer ``Mac control disabled: <check>``; once it is green again
it resolves the ``mac_control`` · ``doctor`` problem an earlier run reported.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import plistlib
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from .bridge import (
    BridgeController,
    Capability,
    Deadline,
    Mutation,
    Problem,
    Refusal,
    before_mutation,
    log_tool,
)
from .bridge_text import compact, normalise_host
from .mac_sites import (
    JS_DISABLED_HINT,
    MEDIA_COMMANDS,
    MediaDriver,
    Outcome,
    Page,
    RealClick,
    Runner,
    Safari,
    SafariError,
    failed,
    ok,
    osascript,
    refused,
    run_argv,
)

log = logging.getLogger("my_stt_tts.mac_control")

FLAG = "MAC_VOICE_MAC_CONTROL"
OPEN = "/usr/bin/open"
MDLS = "/usr/bin/mdls"
FAST_BUDGET_S = 3.0  # list / inspect / fast actions (7.7)
MIN_TIMEOUT_S = 0.5
MAX_URL = 2000
URL_SHORTCUTS = {"youtube": "https://www.youtube.com", "jellyfin": "https://jellyfin.dom42.space"}
VOLUME_STEP = 10
VOLUME_TOLERANCE = 3  # macOS quantises the output volume
BRIGHTNESS_PRESSES = 2  # one press = 1/16
KEY_BRIGHTER, KEY_DARKER = 144, 145
BRIGHTNESS_SETTLE_S = 0.6  # the HUD animates the change
MAX_CANDIDATES = 4
TOOL_NAMES = ("open_url", "open_app", "set_volume", "set_brightness", "media", "youtube_play_first")


def app_dirs() -> tuple[Path, ...]:
    return (Path("/Applications"), Path("/System/Applications"), Path.home() / "Applications")


# -- results -------------------------------------------------------------------------------
def speak(outcome: Outcome | Refusal) -> str:
    """The tool's text for the agent: ``ok: …`` / ``refused: …`` / ``failed: …``."""
    if isinstance(outcome, Refusal):
        return f"refused: {outcome.reason}"
    return f"{outcome.status}: {outcome.message}"


# -- URLs ------------------------------------------------------------------------------------
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_HOST = re.compile(r"^[a-z0-9]([a-z0-9\-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9\-]*[a-z0-9])?)*$")


def _has_foreign_scheme(raw: str) -> bool:
    """``javascript:…`` / ``mailto:…`` but not ``host.tld:8080`` or ``localhost:3000``."""
    match = _SCHEME.match(raw)
    if match is None or "." in match.group(1):
        return False
    rest = raw[match.end() :]
    return not rest[:1].isdigit()


def _ascii_host(host: str) -> str | None:
    try:
        ascii_host = host.encode("idna").decode("ascii").casefold()
    except UnicodeError:
        return None
    return ascii_host if _HOST.fullmatch(ascii_host) else None


def _text_problem(target: Any) -> Refusal | None:
    if not isinstance(target, str) or not target.strip():
        return Refusal("bad_url", "no website named")
    if len(target) > MAX_URL:
        return Refusal("bad_url", "that address is too long")
    if _CONTROL.search(target):
        return Refusal("bad_url", "that address has control characters")
    return None


def _with_scheme(raw: str) -> str | Refusal:
    """``raw`` with an http(s) scheme: bare hosts get ``https://``, other schemes refused."""
    if any(ch.isspace() for ch in raw):
        return Refusal("bad_url", "that address has spaces")
    if "://" in raw:
        scheme_ok = raw.split("://", 1)[0].casefold() in ("http", "https")
    else:
        scheme_ok = not _has_foreign_scheme(raw)
        host_part = raw.split("/", 1)[0]
        bare_word = "." not in host_part and ":" not in host_part
        raw = "https://" + (normalise_host(raw) if bare_word else raw)
    return raw if scheme_ok else Refusal("bad_url", "only web addresses (http or https)")


def normalise_url(target: str) -> str | Refusal:
    """``target`` as a safe https/http URL (7.3 rules), or the refusal."""
    problem = _text_problem(target)
    if problem is not None:
        return problem
    raw = target.strip()
    shortcut = URL_SHORTCUTS.get(compact(raw))
    if shortcut:
        return shortcut
    url = _with_scheme(raw)
    return url if isinstance(url, Refusal) else _checked_url(url)


def _checked_url(url: str) -> str | Refusal:
    try:
        parts = urlsplit(url)
        _ = parts.port  # raises on a malformed port
    except ValueError:
        return Refusal("bad_url", "that is not a valid address")
    if "@" in parts.netloc or parts.username or parts.password:
        return Refusal("bad_url", "addresses with credentials are not allowed")
    host = _ascii_host(parts.hostname or "")
    if host is None:
        return Refusal("bad_url", "that is not a valid address")
    return url


# -- apps ------------------------------------------------------------------------------------
def installed_apps(dirs: Iterable[Path]) -> dict[str, Path]:
    """App name → bundle path (first directory wins), including one level of folders."""
    apps: dict[str, Path] = {}
    for base in dirs:
        for path in _app_bundles(base):
            apps.setdefault(path.stem, path)
    return apps


def _app_bundles(base: Path) -> list[Path]:
    found: list[Path] = []
    with contextlib.suppress(OSError):
        for entry in sorted(base.iterdir()):
            if entry.suffix == ".app":
                found.append(entry)
            elif entry.is_dir() and not entry.name.startswith("."):
                with contextlib.suppress(OSError):
                    found.extend(sorted(p for p in entry.iterdir() if p.suffix == ".app"))
    return found


def _app_key(name: str) -> str:
    name = unicodedata.normalize("NFC", name.strip())
    return compact(name.removesuffix(".app"))


def resolve_app(name: str, apps: Mapping[str, Path]) -> Path | list[str] | None:
    """The app ``name`` means; several candidates → their names; nothing → None."""
    want = _app_key(name)
    if not want:
        return None
    exact = [stem for stem in apps if _app_key(stem) == want]
    if len(exact) == 1:
        return apps[exact[0]]
    partial = sorted(stem for stem in apps if want in _app_key(stem))
    if len(partial) == 1:
        return apps[partial[0]]
    return partial or None


_BUNDLE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-]*$")


def bundle_id(app: Path, runner: Runner, timeout: float) -> str | None:
    """``CFBundleIdentifier`` from Info.plist, else Launch Services via ``mdls``."""
    with contextlib.suppress(OSError, ValueError, plistlib.InvalidFileException):
        with (app / "Contents" / "Info.plist").open("rb") as fh:
            value = plistlib.load(fh).get("CFBundleIdentifier")
        if isinstance(value, str) and _BUNDLE_ID.fullmatch(value):
            return value
    result = runner([MDLS, "-name", "kMDItemCFBundleIdentifier", "-raw", str(app)], timeout)
    value = result.stdout.strip()
    if result.returncode == 0 and _BUNDLE_ID.fullmatch(value) and value != "(null)":
        return value
    return None


# -- display ---------------------------------------------------------------------------------
class Display(Protocol):
    """The main display: built in? brightness 0..1?"""

    def builtin_is_main(self) -> bool: ...

    def brightness(self) -> float | None: ...


class CtypesDisplay:
    """CoreGraphics + private DisplayServices (spike S-BRIGHT); loaded on first use."""

    def __init__(self) -> None:
        self._cg: Any = None
        self._ds: Any = None

    def _load(self) -> None:
        if self._cg is None:
            cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
            cg.CGMainDisplayID.restype = ctypes.c_uint32
            cg.CGDisplayIsBuiltin.restype = ctypes.c_bool
            self._ds = ctypes.CDLL(
                "/System/Library/PrivateFrameworks/DisplayServices.framework/DisplayServices"
            )
            self._cg = cg

    def builtin_is_main(self) -> bool:
        try:
            self._load()
            return bool(self._cg.CGDisplayIsBuiltin(self._cg.CGMainDisplayID()))
        except OSError:
            return False

    def brightness(self) -> float | None:
        try:
            self._load()
            value = ctypes.c_float(0.0)
            rc = self._ds.DisplayServicesGetBrightness(
                self._cg.CGMainDisplayID(), ctypes.byref(value)
            )
        except (OSError, AttributeError):
            return None
        return round(float(value.value), 4) if rc == 0 else None


def ax_trusted() -> bool:
    """Accessibility permission of this process (iTerm's when started from iTerm)."""
    try:
        lib = ctypes.CDLL(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return bool(lib.AXIsProcessTrusted())
    except (OSError, AttributeError):
        return False


# -- AppleScript -----------------------------------------------------------------------------
READ_VOLUME = "output volume of (get volume settings)"
SET_VOLUME = """on run argv
set volume output volume ((item 1 of argv) as integer)
set volume without output muted
return output volume of (get volume settings)
end run"""
SET_MUTE = """on run argv
if (item 1 of argv) is "true" then
set volume with output muted
else
set volume without output muted
end if
return output muted of (get volume settings)
end run"""
PRESS_KEY = """on run argv
tell application "System Events"
repeat ((item 2 of argv) as integer) times
key code ((item 1 of argv) as integer)
delay 0.05
end repeat
end tell
end run"""


def _int_or_none(text: str) -> int | None:
    text = text.strip()
    return int(text) if text.lstrip("-").isdigit() else None


def _signed_number(step: Any) -> int | None:
    """``"+20"`` / ``-3`` / ``5`` → that non-zero whole number, else None."""
    if isinstance(step, bool):
        return None
    if isinstance(step, int | float):
        return int(step) if float(step).is_integer() and step else None
    return _int_or_none(str(step).strip().replace("+", "", 1)) or None


_UP = ("up", "louder", "brighter", "+")
_DOWN = ("down", "quieter", "darker", "-")


def _direction(step: Any) -> str | None:
    text = step.strip().casefold() if isinstance(step, str) else ""
    if text in _UP:
        return "up"
    return "down" if text in _DOWN else None


@dataclass(frozen=True)
class Step:
    """A parsed step: a direction keyword or a signed number (``said`` = what must be said)."""

    delta: int
    said: dict[str, Any]


def parse_step(step: Any, default: int) -> Step | None:
    """``up`` / ``down`` → ±default (keyword must be said), ±N → ±N (number must be said)."""
    direction = _direction(step)
    if direction is not None:
        return Step(default if direction == "up" else -default, {"step": direction})
    number = _signed_number(step)
    if not number:
        return None
    return Step(number, {"step": str(abs(number))})


# -- doctor ----------------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    """One doctor check: ``status`` ok | fail | skip."""

    name: str
    status: str
    detail: str = ""
    fix: str = ""

    @property
    def line(self) -> str:
        if self.status == "ok":
            return f"✅ {self.name}"
        mark = "❌" if self.status == "fail" else "⚠️ "
        text = f"{mark} {self.name}" + (f" — {self.detail}" if self.detail else "")
        return text + (f"\n   fix: {self.fix}" if self.fix else "")


FIX_SYSTEM_EVENTS = (
    "System Settings → Privacy & Security → Automation → iTerm → turn on System Events"
)
FIX_ACCESSIBILITY = "System Settings → Privacy & Security → Accessibility → turn on iTerm"
FIX_SAFARI = "System Settings → Privacy & Security → Automation → iTerm → turn on Safari"
FIX_SAFARI_JS = (
    "Safari → Settings → Advanced → 'Show features for web developers', then"
    " Develop → 'Allow JavaScript from Apple Events'"
)
FIX_VOLUME = "System Settings → Sound → pick an output device that has a volume"
SAFARI_WINDOWS = """if application "Safari" is running then
tell application "Safari" to return (count of windows) as text
end if
return "not running"
"""
SAFARI_JS_PROBE = (  # String(): a bare number comes back as "2.0"
    'tell application "Safari" to do JavaScript "String(1+1)" in current tab of front window'
)


@dataclass
class Doctor:
    """The Mac checks behind ``mac-voice -D`` and :func:`register`."""

    runner: Runner = run_argv
    ax: Callable[[], bool] = ax_trusted
    sleep: Callable[[float], None] = time.sleep
    launch_safari: bool = False  # CLI: start Safari in the background to test it
    timeout: float = 5.0

    def run(self) -> list[Check]:
        safari = self._safari()
        checks = [self._system_events(), self._accessibility(), safari]
        return [*checks, self._safari_js(safari), self._volume()]

    def _osa(self, script: str) -> Any:
        return osascript(self.runner, script, timeout=self.timeout)

    def _system_events(self) -> Check:
        name = "System Events automation"
        res = self._osa(
            'tell application "System Events" to get name of first process whose frontmost is true'
        )
        if res.returncode == 0:
            return Check(name, "ok")
        return Check(name, "fail", _osa_detail(res.stderr), FIX_SYSTEM_EVENTS)

    def _accessibility(self) -> Check:
        name = "Accessibility (key presses)"
        if self.ax():
            return Check(name, "ok")
        return Check(name, "fail", "not trusted", FIX_ACCESSIBILITY)

    def _safari(self) -> Check:
        name = "Safari automation"
        res = self._osa(SAFARI_WINDOWS)
        if (
            res.returncode == 0
            and res.stdout.strip() in ("not running", "0")
            and self.launch_safari
        ):
            res = self._launch_safari()
        if res.returncode != 0:
            return Check(name, "fail", _osa_detail(res.stderr), FIX_SAFARI)
        if res.stdout.strip() == "not running":
            return Check(name, "skip", "Safari is not running — checked on first use")
        return Check(name, "ok", detail=f"windows: {res.stdout.strip()}")

    def _launch_safari(self) -> Any:
        self.runner([OPEN, "-g", "-a", "Safari"], self.timeout)
        res = self._osa(SAFARI_WINDOWS)
        for _ in range(10):
            if res.returncode != 0 or res.stdout.strip() not in ("not running", "0"):
                break
            self.sleep(0.5)
            res = self._osa(SAFARI_WINDOWS)
        return res

    def _safari_js(self, safari: Check) -> Check:
        name = "Safari 'Allow JavaScript from Apple Events'"
        if safari.status != "ok":
            return Check(name, "skip", "needs Safari automation and an open window")
        if safari.detail == "windows: 0":
            return Check(name, "skip", "no Safari window to test in — checked on first use")
        res = self._osa(SAFARI_JS_PROBE)
        if res.returncode == 0 and res.stdout.strip() == "2":
            return Check(name, "ok")
        detail = _osa_detail(res.stderr) if res.returncode else "unexpected reply"
        return Check(name, "fail", detail, FIX_SAFARI_JS)

    def _volume(self) -> Check:
        name = "Volume read"
        res = self._osa(READ_VOLUME)
        if res.returncode == 0 and _int_or_none(res.stdout) is not None:
            return Check(name, "ok")
        detail = _osa_detail(res.stderr) if res.returncode else "this output has no volume"
        return Check(name, "fail", detail, FIX_VOLUME)


def _osa_detail(stderr: str) -> str:
    text = " ".join(stderr.split())
    match = re.search(r"\((-?\d+)\)\s*$", text)
    if JS_DISABLED_HINT in text:
        return "JavaScript from Apple Events is off"
    if match and match.group(1) == "-1743":
        return "not authorised (-1743)"
    if match and match.group(1) == "-1719":
        return "no assistive access (-1719)"
    return text[-120:] or "failed"


def doctor_main(runner: Runner = run_argv, ax: Callable[[], bool] = ax_trusted) -> int:
    """``mac-voice -D``: print one ✅/❌ line per check; exit 0 when all are green."""
    checks = Doctor(runner=runner, ax=ax, launch_safari=True).run()
    for check in checks:
        print(check.line)
    green = all(c.status == "ok" for c in checks)
    print("✅ Mac control ready" if green else "❌ Mac control stays disabled until all are ✅")
    return 0 if green else 1


# -- the tools -------------------------------------------------------------------------------
class MacControl:  # pylint: disable=too-many-instance-attributes  # the fast path's wiring
    """The fast-path tools bound to one controller."""

    def __init__(
        self,
        controller: BridgeController,
        *,
        runner: Runner = run_argv,
        display: Display | None = None,
        apps: Callable[[], Iterable[Path]] = app_dirs,
        sleep: Callable[[float], None] = time.sleep,
        real_click: RealClick | None = None,
    ) -> None:
        self.controller = controller
        self.runner = runner
        self.display: Display = display if display is not None else CtypesDisplay()
        self.apps = apps
        self.sleep = sleep
        self.media_driver = MediaDriver(
            Safari(runner), sleep=sleep, clock=controller.clock, real_click=real_click
        )
        self.disabled: str | None = None

    @property
    def tools(self) -> dict[str, Callable[..., str]]:
        return {name: getattr(self, name) for name in TOOL_NAMES}

    # -- shared mechanics ----------------------------------------------------------------
    def _deadline(self) -> Deadline:
        return Deadline(FAST_BUDGET_S, clock=self.controller.clock)

    @staticmethod
    def _timeout(deadline: Deadline) -> float:
        return max(MIN_TIMEOUT_S, deadline.remaining())

    def _start(self, tool: str, detail: str, args: Mapping[str, Any]) -> Capability | str:
        """Log, mint, and check that ``args`` come from the transcript (not consumed yet)."""
        log_tool(tool, detail)
        cap = self.controller.mint()
        if isinstance(cap, Refusal):
            return speak(cap)
        refusal = self.controller.caps.check(cap, tool, args)
        return speak(refusal) if refusal is not None else cap

    def _mutate(
        self,
        mutation: Mutation,
        deadline: Deadline,
        act: Callable[[], Outcome],
    ) -> str:
        """Gate (lock, cancellation, deadline, capability) → act → report failures."""
        token = self.controller.cancel
        if not self.controller.mutation_lock.acquire(timeout=deadline.remaining()):
            return speak(refused("another action is still running"))
        try:
            gate = before_mutation(token, deadline, self.controller.caps, mutation)
            if gate is not None:
                return speak(gate)
            outcome = act()
        finally:
            self.controller.mutation_lock.release()
        return self._finish(mutation.action, outcome)

    def _finish(self, subject: str, outcome: Outcome) -> str:
        if outcome.status == "failed":
            self.controller.problems.report(Problem("mac_action", subject, outcome.message))
        return speak(outcome)

    def _fail(self, subject: str, message: str) -> str:
        return self._finish(subject, failed(message))

    def _run(self, argv: list[str], timeout: float, what: str) -> Outcome:
        result = self.runner(argv, timeout)
        if result.returncode == 0:
            return ok(what)
        if result.returncode == 124:
            return failed(f"{what}: took too long")
        return failed(f"could not {what}")

    # -- open_url ------------------------------------------------------------------------
    def open_url(self, target: str) -> str:
        """Open any http(s) website in Safari (``youtube`` / ``jellyfin`` shortcuts)."""
        url = normalise_url(target)
        if isinstance(url, Refusal):
            log_tool("open_url", "?")
            return speak(url)
        host = urlsplit(url).hostname or ""
        start = self._start("open_url", host, {"target": str(target)})
        if isinstance(start, str):
            return start
        deadline = self._deadline()
        argv = [OPEN, "-a", "Safari", url]
        return self._mutate(
            Mutation("open_url", {"target": str(target)}, start),
            deadline,
            lambda: self._run(argv, self._timeout(deadline), f"open {host}"),
        )

    # -- open_app ------------------------------------------------------------------------
    def open_app(self, name: str) -> str:
        """Open an installed app by name (several matches → the candidates)."""
        start = self._start("open_app", str(name)[:40], {"name": str(name)})
        if isinstance(start, str):
            return start
        deadline = self._deadline()
        found = resolve_app(str(name), installed_apps(self.apps()))
        if found is None:
            return speak(refused(f"no app called {name}"))
        if isinstance(found, list):
            names = ", ".join(found[:MAX_CANDIDATES])
            return speak(refused(f"which one: {names}?"))
        bid = bundle_id(found, self.runner, self._timeout(deadline))
        if bid is None:
            return self._fail("open_app", f"{found.stem} has no bundle id")
        argv = [OPEN, "-b", bid]
        return self._mutate(
            Mutation("open_app", {"name": str(name)}, start),
            deadline,
            lambda: self._run(argv, self._timeout(deadline), f"open {found.stem}"),
        )

    # -- volume --------------------------------------------------------------------------
    def set_volume(
        self, level: int | str | None = None, step: int | str | None = None, mute: Any = None
    ) -> str:
        """Set the output volume: ``level`` 0–100, ``step`` up/down/±N, or ``mute``."""
        plan = _volume_plan(level, step, mute)
        if isinstance(plan, Refusal):
            log_tool("set_volume", "?")
            return speak(plan)
        detail, said, apply = plan
        start = self._start("set_volume", detail, said)
        if isinstance(start, str):
            return start
        deadline = self._deadline()
        if apply[0] == "mute":
            return self._mutate(
                Mutation("set_volume", said, start),
                deadline,
                lambda: self._mute(bool(apply[1]), deadline),
            )
        current = self._read_volume(deadline)
        if isinstance(current, Outcome):
            return self._finish("set_volume", current)
        target = _volume_target(apply, current)
        if isinstance(target, Outcome):
            return speak(target)
        return self._mutate(
            Mutation("set_volume", said, start),
            deadline,
            lambda: self._apply_volume(target, current, deadline),
        )

    def _read_volume(self, deadline: Deadline) -> int | Outcome:
        res = osascript(self.runner, READ_VOLUME, timeout=self._timeout(deadline))
        value = _int_or_none(res.stdout) if res.returncode == 0 else None
        if value is None:
            return failed(
                "cannot read the volume" if res.returncode else "no volume on this output"
            )
        return value

    def _apply_volume(self, target: int, before: int, deadline: Deadline) -> Outcome:
        res = osascript(self.runner, SET_VOLUME, str(target), timeout=self._timeout(deadline))
        after = _int_or_none(res.stdout) if res.returncode == 0 else None
        if after is None:
            return failed("could not set the volume")
        moved_right = (target > before and after > before) or (target < before and after < before)
        if abs(after - target) <= VOLUME_TOLERANCE or moved_right:
            return ok(f"volume {after}")
        return failed(f"volume stayed at {after}")

    def _mute(self, on: bool, deadline: Deadline) -> Outcome:
        flag = "true" if on else "false"
        res = osascript(self.runner, SET_MUTE, flag, timeout=self._timeout(deadline))
        if res.returncode == 0 and res.stdout.strip() == flag:
            return ok("muted" if on else "unmuted")
        return failed("could not mute" if on else "could not unmute")

    # -- brightness ----------------------------------------------------------------------
    def set_brightness(self, step: int | str) -> str:
        """Brighter / darker on the built-in display (``up`` / ``down`` or ±N presses)."""
        parsed = parse_step(step, BRIGHTNESS_PRESSES)
        if parsed is None or abs(parsed.delta) > 16:
            log_tool("set_brightness", "?")
            return speak(refused("say brighter or darker"))
        start = self._start("set_brightness", f"{parsed.delta:+d}", parsed.said)
        if isinstance(start, str):
            return start
        if not self.display.builtin_is_main():
            return speak(refused("the main display is external — ask the Mac operator"))
        before = self.display.brightness()
        if before is None:
            return self._fail("set_brightness", "cannot read the brightness")
        if (parsed.delta > 0 and before >= 1.0) or (parsed.delta < 0 and before <= 0.0):
            return speak(ok("brightness already at the limit"))
        deadline = self._deadline()
        return self._mutate(
            Mutation("set_brightness", parsed.said, start),
            deadline,
            lambda: self._press_brightness(parsed.delta, before, deadline),
        )

    def _press_brightness(self, delta: int, before: float, deadline: Deadline) -> Outcome:
        code = KEY_BRIGHTER if delta > 0 else KEY_DARKER
        res = osascript(
            self.runner, PRESS_KEY, str(code), str(abs(delta)), timeout=self._timeout(deadline)
        )
        if res.returncode != 0:
            detail = _osa_detail(res.stderr)
            return failed(f"key presses were refused ({detail}) — run mac-voice -D")
        self.sleep(BRIGHTNESS_SETTLE_S)
        after = self.display.brightness()
        if after is not None and ((after > before) if delta > 0 else (after < before)):
            return ok(f"brightness {round(after * 100)} percent")
        return failed("the brightness did not change")

    # -- media -------------------------------------------------------------------------
    def media(self, command: str, seconds: float | int | str | None = None) -> str:
        """play | pause | seek (±seconds) | next | previous | fullscreen in the front Safari."""
        cmd = str(command).strip().casefold()
        said = _media_said(cmd, seconds)
        if isinstance(said, Refusal):
            log_tool("media", "?")
            return speak(said)
        start = self._start("media", cmd, said)
        if isinstance(start, str):
            return start
        deadline = self._deadline()
        page = self._media_page(cmd, deadline)
        if isinstance(page, str):
            return page
        delta = float(seconds) if seconds is not None else 10.0
        return self._mutate(
            Mutation("media", said, start),
            deadline,
            lambda: self.media_driver.act(page, cmd, delta),
        )

    def _media_page(self, cmd: str, deadline: Deadline) -> Page | str:
        """Inspect + precheck the front window (refusals and no-ops as tool text)."""
        try:
            page = self.media_driver.inspect(self._timeout(deadline))
            if isinstance(page, Outcome):
                return speak(page)
            pre = self.media_driver.precheck(page, cmd, self._timeout(deadline))
        except SafariError as exc:
            return self._fail(f"media {cmd}", exc.reason)
        return speak(pre) if pre is not None else page

    def youtube_play_first(self) -> str:
        """Open and play the first visible video of the YouTube page in the front window."""
        start = self._start("youtube_play_first", "", {})
        if isinstance(start, str):
            return start
        deadline = self._deadline()
        try:
            page = self.media_driver.inspect(self._timeout(deadline))
            url = page if isinstance(page, Outcome) else self.media_driver.first_tile(page)
        except SafariError as exc:
            return self._fail("youtube_play_first", exc.reason)
        if isinstance(url, Outcome):
            return speak(url)
        assert isinstance(page, Page)
        return self._mutate(
            Mutation("youtube_play_first", {}, start),
            deadline,
            lambda: self.media_driver.play_first(page, url),
        )

    # -- disabled stubs --------------------------------------------------------------------
    def disabled_tool(self, name: str) -> Callable[..., str]:
        reason = self.disabled or "unknown"

        def _disabled(*_args: Any, **_kwargs: Any) -> str:
            log_tool(name, "disabled")
            return f"Mac control disabled: {reason}"

        return _disabled


def _volume_plan(
    level: Any, step: Any, mute: Any
) -> tuple[str, dict[str, Any], tuple[str, Any]] | Refusal:
    """(log detail, arguments that must be derivable, what to apply) or a refusal."""
    given = [x is not None for x in (level, step, mute)]
    if sum(given) != 1:
        return Refusal("bad_args", "say a volume level, louder / quieter, or mute")
    if mute is not None:
        on = mute if isinstance(mute, bool) else str(mute).strip().casefold() in ("true", "1", "on")
        return ("mute" if on else "unmute", {"mute" if on else "unmute": True}, ("mute", on))
    if level is not None:
        value = _int_or_none(str(level))
        if value is None or not 0 <= value <= 100:
            return Refusal("bad_args", "the volume goes from 0 to 100")
        return (str(value), {"level": value}, ("level", value))
    parsed = parse_step(step, VOLUME_STEP)
    if parsed is None or abs(parsed.delta) > 100:
        return Refusal("bad_args", "say louder or quieter")
    return (f"{parsed.delta:+d}", parsed.said, ("step", parsed.delta))


def _volume_target(apply: tuple[str, Any], current: int) -> int | Outcome:
    kind, value = apply
    if kind == "level":
        return int(value)
    target = min(100, max(0, current + int(value)))
    if target == current:
        return ok(f"volume already {current}")
    return target


def _media_said(cmd: str, seconds: Any) -> dict[str, Any] | Refusal:
    """The media arguments that must come from the transcript."""
    if cmd not in MEDIA_COMMANDS:
        return Refusal("bad_args", f"unknown media command {cmd[:20]}")
    if cmd != "seek":
        return {"command": cmd}
    if seconds is None:
        return {"command": "seek"}
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return Refusal("bad_args", "say how many seconds")
    if not value.is_integer() or not 0 < abs(value) <= 3600:
        return Refusal("bad_args", "say how many seconds")
    return {"seconds": str(abs(int(value)))}


# -- registration ----------------------------------------------------------------------------
def register(
    controller: BridgeController,
    runner: Runner | None = None,
    env: Mapping[str, str] | None = None,
    **options: Any,
) -> MacControl | None:
    """Register the Mac tools on ``controller`` when ``MAC_VOICE_MAC_CONTROL=1``.

    ``options`` go to :class:`MacControl` (``display``, ``apps``, ``sleep``,
    ``real_click``) plus ``ax`` for the doctor. While a doctor check fails, every tool
    name is registered as a stub answering ``Mac control disabled: <check>`` and one
    problem is reported; a green doctor resolves that problem. Returns the
    :class:`MacControl` (None when the flag is off).
    """
    env = os.environ if env is None else env
    if env.get(FLAG, "").strip() != "1":
        return None
    runner = runner or run_argv
    ax = options.pop("ax", ax_trusted)
    mac = MacControl(controller, runner=runner, **options)
    doctor = Doctor(runner=runner, ax=ax, sleep=mac.sleep)
    bad = [c for c in doctor.run() if c.status == "fail"]
    if bad and any(c.detail == "timeout" for c in bad):
        mac.sleep(1.0)  # a busy Safari can miss one 5 s Apple-Event timeout; ask once more
        bad = [c for c in doctor.run() if c.status == "fail"]
    if bad:
        mac.disabled = bad[0].name
        log.warning("⚠️  Mac control disabled: %s (run mac-voice -D)", mac.disabled)
        controller.problems.report(Problem("mac_control", "doctor", f"disabled: {mac.disabled}"))
        for name in TOOL_NAMES:
            controller.register_tool(name, mac.disabled_tool(name))
        return mac
    for name, fn in mac.tools.items():
        controller.register_tool(name, fn)
    log.info("🖥️  Mac control on (%d tools)", len(TOOL_NAMES))
    try:  # a green doctor settles a "disabled" problem an earlier run left in the inbox
        controller.problems.resolve("mac_control", "doctor")
    except Exception:  # pylint: disable=broad-exception-caught  # the tools stay registered
        log.warning("⚠️  could not resolve the old Mac control problem", exc_info=True)
    return mac
